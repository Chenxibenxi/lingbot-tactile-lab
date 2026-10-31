"""Opt-in, read-only SAPIEN gripper contact telemetry. Not a GelSight sensor.

Only contact impulses from the just-completed physics step are treated as physical
measurements. Zero-impulse near-contacts are not counted as touches. This module
never changes robot drive targets, success checks, model observations or seeds.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


def _name(body):
    return str(body.entity.name)


def _xyz(value):
    try:
        values = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return values if len(values) == 3 and all(math.isfinite(x) for x in values) else None


def gripper_links(robot):
    """Resolve actual finger *link* names; fail closed on ambiguous left/right links."""
    result = {}
    for side in ("left", "right"):
        joints = getattr(robot, f"{side}_gripper")
        links = {joint.child_link.get_name() for joint, *_ in joints}
        links.update(getattr(robot, f"{side}_fix_gripper_name", []))
        if not links:
            raise ValueError(f"No {side} gripper collision links resolved")
        result[side] = links
    shared = result["left"] & result["right"]
    if shared:
        raise ValueError(f"Ambiguous gripper links: {sorted(shared)}")
    return result


def sample_contacts(scene, links, dt, *, min_impulse=1e-9, robot_links=()):
    """Return per-arm summed |impulse|/dt proxy plus signed world-frame impulse.

    SAPIEN documents point.impulse as impulse *on contact.bodies[0]*; sign is
    reversed for bodies[1]. Magnitudes sum over finger points to avoid opposite
    fingers canceling, but MUST NOT be interpreted as calibrated grasp normal N.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("positive physics timestep required")
    labels = {name: side for side, names in links.items() for name in names}
    sides = {side: {"contacts": 0, "points": 0, "impulse_magnitude_ns": 0.0,
                    "load_proxy_n": 0.0, "net_impulse_world_ns": [0.0, 0.0, 0.0],
                    "objects": [], "fingers": {
                        name: {"contacts": 0, "points": 0, "impulse_magnitude_ns": 0.0,
                               "load_proxy_n": 0.0, "objects": []}
                        for name in sorted(links[side])}}
             for side in ("left", "right")}
    objects = {side: set() for side in sides}
    finger_objects = {name: set() for names in links.values() for name in names}
    robot_links = set(robot_links)
    # Avoid collecting privileged contact geometry as a model input; telemetry only.
    for contact in scene.get_contacts():
        a, b = (_name(body) for body in contact.bodies)
        side_a, side_b = labels.get(a), labels.get(b)
        if bool(side_a) == bool(side_b):  # neither finger or two gripper links
            continue
        side = side_a or side_b
        other = b if side_a else a
        if other in robot_links:  # gripper-camera/arm self-contact is not touch
            continue
        record = sides[side]
        finger_name = a if side_a else b
        finger_record = record["fingers"][finger_name]
        active = 0
        for point in contact.points:
            impulse = _xyz(getattr(point, "impulse", None))
            if impulse is None:
                raise RuntimeError("SAPIEN contact point lacks finite impulse; cannot estimate force")
            magnitude = math.sqrt(sum(x * x for x in impulse))
            if magnitude <= min_impulse:
                continue
            active += 1
            record["points"] += 1
            record["impulse_magnitude_ns"] += magnitude
            finger_record["points"] += 1
            finger_record["impulse_magnitude_ns"] += magnitude
            sign = 1.0 if side_a else -1.0
            for i in range(3):
                record["net_impulse_world_ns"][i] += sign * impulse[i]
        if active:
            record["contacts"] += 1
            finger_record["contacts"] += 1
            objects[side].add(other)
            finger_objects[finger_name].add(other)
    for side, record in sides.items():
        record["load_proxy_n"] = record["impulse_magnitude_ns"] / dt
        record["objects"] = sorted(objects[side])
        for finger_name, finger_record in record["fingers"].items():
            finger_record["load_proxy_n"] = finger_record["impulse_magnitude_ns"] / dt
            finger_record["objects"] = sorted(finger_objects[finger_name])
    return sides


class ForceProbe:
    """Stream one JSONL per episode, no GPU allocation or video rendering."""

    def __init__(self, output_dir, *, task, seed, episode, dt, robot):
        self.links = gripper_links(robot)
        self.robot_links = set()
        for entity in (getattr(robot, "left_entity", None), getattr(robot, "right_entity", None)):
            if entity is not None:
                self.robot_links.update(link.get_name() for link in entity.get_links())
        self.dt = float(dt)
        if self.dt <= 0 or not math.isfinite(self.dt):
            raise ValueError("Invalid scene timestep")
        self.output_dir = Path(output_dir)
        self.task, self.seed, self.episode = str(task), int(seed), int(episode)
        self.step = 0
        self.last = None
        self._file = None
        self._obs_file = None
        self._frame_file = None
        self._policy_file = None
        self._policy_index = 0
        self._obs_index = 0
        self._last_observation_step = None
        self._frame_count = 0
        self.motion_index = 0
        self.motion_phase = "unclassified"
        self.clock_origin = ("first_instrumented_expert_step_after_setup_stability"
                             if os.environ.get("FORCE_PROBE_EXPERT_DIR") else
                             "first_instrumented_policy_action_step_after_setup")
        self._failed = False
        self._live = None
        if os.environ.get("TACTILE_LIVE_WINDOW") == "1":
            from live_tactile_window import LiveTactileWindow
            self._live = LiveTactileWindow(self.links, self.dt)

    def _ensure_file(self):
        if self._file is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Keep raw physics and camera-event streams paired by a unique basename.
        import tempfile
        self._file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f"{self.task}_{self.seed}_{self.episode}_",
            suffix=".jsonl", dir=self.output_dir, delete=False)
        self._file.write(json.dumps({"schema": "phizju_force_probe_v1", "task": self.task,
            "seed": self.seed, "episode": self.episode, "dt_s": self.dt,
            "clock_origin": self.clock_origin,
            "side_links": {s: sorted(v) for s, v in self.links.items()},
            "meaning": "physx contact impulse/dt proxy; not GelSight; not calibrated grip force"}) + "\n")

    @property
    def failed(self):
        return self._failed

    @property
    def raw_path(self):
        return Path(self._file.name) if self._file is not None else None

    @property
    def saved_frames(self):
        return self._frame_count

    def begin_motion(self, phase):
        if phase not in {"expert_dense", "expert_together", "policy_action"}:
            raise ValueError("invalid motion phase")
        self.motion_index += 1
        self.motion_phase = phase

    def sample(self, scene, robot, action_index):
        if self._failed:
            return
        try:
            data = sample_contacts(scene, self.links, self.dt, robot_links=self.robot_links)
            self._ensure_file()
            self.step += 1
            self.last = {"physics_step": self.step, "sim_time_s": round(self.step * self.dt, 8),
                "action_index": int(action_index), "motion_index": self.motion_index,
                "motion_phase": self.motion_phase, "left": data["left"], "right": data["right"]}
            self._file.write(json.dumps(self.last, allow_nan=False) + "\n")
            if self._live is not None:
                self._live.push(self.last)
            if self.step % 250 == 0:
                self._file.flush()
        except Exception as exc:
            # Probe failures do not alter simulation/evaluation but remain observable.
            self._failed = True
            print(f"FORCE_PROBE_ERROR {type(exc).__name__}: {exc}", flush=True)

    def policy_window(self):
        """Opt-in bounded history; ONLY the experimental client may attach at infer."""
        if self._failed:
            raise RuntimeError("failed tactile probe cannot be used for inference")
        if self._live is None:
            return None
        return self._live.snapshot(task=self.task, seed=self.seed, episode=self.episode)

    def record_observation(self, observation, action_index):
        """Log camera-capture boundary, not a claim that policy inference happened.

        The first marker may be step=0 because setup physics was not instrumented.
        Never infer frame time from LeRobot's packaging FPS or from frame index.
        """
        if self._failed:
            return
        try:
            self._ensure_file()
            if self._obs_file is None:
                self._obs_file = open(Path(self._file.name).with_suffix(".obs.jsonl"), "x", encoding="utf-8")
                self._obs_file.write(json.dumps({"schema": "phizju_obs_marker_v1",
                    "probe_file": Path(self._file.name).name, "task": self.task,
                    "seed": self.seed, "episode": self.episode, "dt_s": self.dt,
                    "kind": "get_obs_camera_capture; not necessarily policy_request"}) + "\n")
            cameras = {}
            for name in ("head_camera", "left_camera", "right_camera"):
                rgb = observation.get("observation", {}).get(name, {}).get("rgb")
                cameras[name] = list(map(int, rgb.shape)) if rgb is not None else None
            self._obs_file.write(json.dumps({"obs_index": self._obs_index,
                "physics_step": self.step, "sim_time_s": round(self.step * self.dt, 8),
                "action_index": int(action_index), "valid": self.last is not None,
                "capture_monotonic_ns": time.monotonic_ns(), "rgb_shapes": cameras},
                allow_nan=False) + "\n")
            self._obs_index += 1
            self._last_observation_step = self.step
            self._obs_file.flush()
        except Exception as exc:
            self._failed = True
            print(f"FORCE_PROBE_OBS_ERROR {type(exc).__name__}: {exc}", flush=True)

    def record_policy_request(self):
        """Marker ONLY when the experimental client is about to call infer."""
        if self._failed or self._obs_index <= 0 or self._last_observation_step != self.step:
            raise RuntimeError("cannot record an unobserved, stale or failed policy request")
        self._ensure_file()
        if self._policy_file is None:
            self._policy_file = open(Path(self._file.name).with_suffix(".policy.jsonl"), "x", encoding="utf-8")
            self._policy_file.write(json.dumps({"schema":"phizju_policy_request_v1",
                "probe_file":Path(self._file.name).name, "task":self.task,
                "seed":self.seed, "episode":self.episode, "dt_s":self.dt,
                "kind":"client_infer_call_attempt; transport_ack_unverified"}) + "\n")
        self._policy_file.write(json.dumps({"request_index":self._policy_index,
            "obs_index":self._obs_index-1, "physics_step":self.step,
            "sim_time_s":round(self.step*self.dt,8),
            "request_monotonic_ns":time.monotonic_ns()},allow_nan=False) + "\n")
        self._policy_file.flush()
        self._policy_index += 1

    def record_saved_frame(self, frame_index, observation, robot):
        """Call only AFTER numbered frame pickle has been saved successfully."""
        if self._failed:
            return
        try:
            if frame_index != self._frame_count or self._obs_index <= 0:
                raise ValueError("saved frame index/observation marker mismatch")
            self._ensure_file()
            if self._frame_file is None:
                self._frame_file = open(Path(self._file.name).with_suffix(".frames.jsonl"), "x", encoding="utf-8")
                self._frame_file.write(json.dumps({"schema": "phizju_saved_frames_v1",
                    "probe_file": Path(self._file.name).name, "task": self.task,
                    "seed": self.seed, "episode": self.episode, "dt_s": self.dt,
                    "clock_origin": "first_instrumented_expert_step_after_setup_stability",
                    "hdf5_expected": f"data/episode{self.episode}.hdf5"}) + "\n")
            state = observation.get("joint_action", {}).get("vector")
            if state is None or len(state) != 14:
                raise ValueError("missing 14D saved command state")
            actual = {}
            for side in ("left", "right"):
                entity = getattr(robot, f"{side}_entity")
                actual[side] = {"names": [joint.get_name() for joint in entity.get_active_joints()],
                    "qpos": entity.get_qpos().tolist(), "qvel": entity.get_qvel().tolist()}
                if len(actual[side]["names"]) != len(actual[side]["qpos"]) or len(actual[side]["names"]) != len(actual[side]["qvel"]):
                    raise ValueError("actual joint state length mismatch")
            row = {"frame_index": int(frame_index), "obs_index": self._obs_index - 1,
                   "physics_step": self.step, "sim_time_s": round(self.step*self.dt, 8),
                   "motion_index": self.motion_index, "motion_phase": self.motion_phase,
                   "command_state_14": state.tolist() if hasattr(state, "tolist") else list(state),
                   "actual_joints": actual}
            self._frame_file.write(json.dumps(row, allow_nan=False) + "\n")
            self._frame_file.flush()
            self._frame_count += 1
        except Exception as exc:
            self._failed = True
            print(f"FORCE_PROBE_FRAME_ERROR {type(exc).__name__}: {exc}", flush=True)

    def observation(self):
        """Optional observation channel; never injected into existing policy RPC."""
        if self.last is None:
            return {"valid": False, "physics_step": 0, "left_load_proxy_n": 0.0,
                    "right_load_proxy_n": 0.0, "left_contact": False, "right_contact": False}
        return {"valid": not self._failed, "physics_step": self.step,
                "left_load_proxy_n": self.last["left"]["load_proxy_n"],
                "right_load_proxy_n": self.last["right"]["load_proxy_n"],
                "left_contact": self.last["left"]["points"] > 0,
                "right_contact": self.last["right"]["points"] > 0}

    def close(self):
        if self._policy_file is not None:
            self._policy_file.close()
            self._policy_file = None
        if self._frame_file is not None:
            self._frame_file.close()
            self._frame_file = None
        if self._obs_file is not None:
            self._obs_file.close()
            self._obs_file = None
        if self._file is not None:
            self._file.close()
            self._file = None

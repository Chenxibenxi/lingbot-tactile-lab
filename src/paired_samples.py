"""CPU paired-sample reader with original physical time and strict relay provenance.

Does NOT pretend LeRobot's packaging fps=50 is a physical camera clock.
``input`` never contains future contact/action; ``target`` is next-frame 14D
command, matching the existing RoboTwin→LeRobot training convention.
"""
from __future__ import annotations

from pathlib import Path

from build_paired_index import build
from register_dataset import build_card

CAMERAS = ("head_camera", "left_camera", "right_camera")


def _raw_files(trial):
    candidates = [p for p in (trial / "force").glob("*.jsonl")
                  if not p.name.endswith((".obs.jsonl", ".frames.jsonl", ".policy.jsonl"))]
    if len(candidates) != 1:
        raise ValueError("expected one same-run force stream")
    return candidates[0]


class PairedEpisode:
    def __init__(self, repo, trial_id, *, instruction, require_valid_history=True,
                 require_positive_frame_delta=True):
        self.repo = Path(repo).resolve(strict=True)
        self.trial = (self.repo / "data/paired" / trial_id).resolve(strict=True)
        self.card = build_card(self.trial, self.repo)  # role, host, SHA256 and HDF5 checks
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("explicit non-empty same-task language instruction required")
        self.instruction = instruction.strip()
        self.raw = _raw_files(self.trial)
        self.hdf5 = self.trial / "demonstration/data" / f"episode{self.card['episode']}.hdf5"
        self.rows = []
        for row in build(self.raw, self.raw.with_suffix(".obs.jsonl"),
                         self.raw.with_suffix(".frames.jsonl"), self.hdf5,
                         self.trial / "completion.json"):
            inp, target = row["input"], row["target"]
            if require_valid_history and not inp["tactile_window_valid"]:
                continue
            if require_positive_frame_delta and target["next_frame_physics_step"] <= inp["capture_physics_step"]:
                continue
            self.rows.append(row)
        if not self.rows:
            raise ValueError("no causal usable paired samples after quality filters")

    def __len__(self):
        return len(self.rows)

    def sample(self, index, *, include_images=True):
        """Loads only current frame images; no future RGB/force in model input."""
        import numpy as np
        row = self.rows[index]
        inp, target = row["input"], row["target"]
        features = np.asarray(inp["tactile_history"], dtype=np.float32)
        mask = np.asarray(inp["tactile_mask"], dtype=np.float32)
        command = np.asarray(inp["command_state_14"], dtype=np.float32)
        action = np.asarray(target["action_command_14"], dtype=np.float32)
        if features.shape != (5, 4, 3) or mask.shape != (5, 4) or command.shape != (14,) or action.shape != (14,):
            raise ValueError("unexpected feature, mask, state or action shape")
        if not np.isfinite(features).all() or not np.isfinite(mask).all() or not np.isfinite(command).all() or not np.isfinite(action).all():
            raise ValueError("nonfinite paired sample")
        if not np.logical_and(mask >= 0, mask <= 1).all():
            raise ValueError("invalid tactile coverage mask")
        age_ms = inp["tactile_quality"].get("force_age_ms")
        if inp["tactile_window_valid"] and (age_ms is None or not np.isfinite(age_ms) or age_ms < 0):
            raise ValueError("valid tactile window lacks finite nonnegative force age")
        result = {"input": {"command_state_14": command,
                            "tactile_history": features,
                            "tactile_mask": mask,
                            "tactile_window_valid": bool(inp["tactile_window_valid"]),
                            "tactile_age_ms": np.float32(age_ms if age_ms is not None else -1.0),
                            "capture_physics_step": inp["capture_physics_step"],
                            "sim_time_s": inp["capture_sim_time_s"],
                            "instruction": self.instruction},
                  "target": {"action_command_14": action},
                  "metadata": {"trial_id": self.trial.name, "task": row["task"],
                               "seed": row["seed"], "frame_index": inp["frame_index"],
                               "target_frame_index": target["action_source_frame_index"],
                               "next_frame_step_delta": target["next_frame_physics_step"] - inp["capture_physics_step"],
                               "tactile_quality": inp["tactile_quality"]}}
        if include_images:
            import cv2
            import h5py
            images = {}
            with h5py.File(self.hdf5, "r") as hdf:
                for cam in CAMERAS:
                    blob = hdf[f"observation/{cam}/rgb"][inp["frame_index"]]
                    bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError(f"failed to decode {cam} frame {inp['frame_index']}")
                    images[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result["input"]["images_rgb"] = images
        return result


def describe(repo, trial_id, instruction):
    episode=PairedEpisode(repo,trial_id,instruction=instruction)
    frames=[row["input"]["frame_index"] for row in episode.rows]
    return {"schema":"lingbot_tactile_paired_readiness_v1",
            "trial_id":trial_id,"source_role":episode.card["source_machine_role"],
            "original_frames":episode.card["saved_frames"],"causal_usable_pairs":len(episode),
            "first_usable_frame":min(frames),"last_usable_frame":max(frames),
            "images_same_run":True,"ready_for_training":False,
            "why_not_yet":"need multi-seed train/val/test split and train-only normalization before model fitting"}

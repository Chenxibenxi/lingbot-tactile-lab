"""Causal, streaming 250-Hz gripper-force windows at RoboTwin camera captures.

Input is ONE episode's E009 physics JSONL and its paired *.obs.jsonl marker file.
No LeRobot synthetic FPS, no interpolation and no future physics steps are used.
This is an observation-candidate dataset, NOT directly trainable VLA pairs.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path

FEATURES = ("mean_load_proxy_n", "peak_load_proxy_n", "contact_fraction")
SIDES = ("left", "right")


def _lines(path):
    with Path(path).open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def _input_header(head, obs_head, source_name):
    if head.get("schema") != "phizju_force_probe_v1" or obs_head.get("schema") != "phizju_obs_marker_v1":
        raise ValueError("unsupported force/observation schema")
    if obs_head.get("probe_file") != source_name:
        raise ValueError("observation markers do not belong to this force file")
    for key in ("task", "seed", "episode"):
        if head[key] != obs_head[key]:
            raise ValueError(f"episode identity mismatch: {key}")
    dt = float(head["dt_s"])
    if not math.isfinite(dt) or dt <= 0 or not math.isclose(dt, float(obs_head["dt_s"]), abs_tol=1e-7):
        raise ValueError("invalid/mismatched physics timestep")
    links = {side: sorted(head["side_links"][side]) for side in SIDES}
    if any(not links[side] for side in SIDES):
        raise ValueError("missing gripper links")
    if set(links["left"]) & set(links["right"]):
        raise ValueError("left/right finger links overlap")
    return head, obs_head, links, dt


def align(raw_path, markers_path, *, window_steps=25, bins=5, peak_warning_n=500):
    """Yield one feature record per camera capture with closed interval [T-W+1,T].

    Raw and marker streams must be monotonically ordered. Missing raw frames
    remain missing (masked), never forward-filled or interpolated. Metadata
    'last_source_step' proves every used sample occurred no later than anchor T.
    """
    if window_steps <= 0 or bins <= 0 or window_steps % bins:
        raise ValueError("window_steps must be a positive multiple of bins")
    if not math.isfinite(peak_warning_n) or peak_warning_n <= 0:
        raise ValueError("peak_warning_n must be positive")
    raw_iter, obs_iter = _lines(raw_path), _lines(markers_path)
    # Bind streams by exact filename as well as episode identity.
    raw_head = next(raw_iter, None); obs_head = next(obs_iter, None)
    if not isinstance(raw_head, dict) or not isinstance(obs_head, dict):
        raise ValueError("missing input header")
    (raw_head, obs_head, links, dt) = _input_header(raw_head, obs_head, Path(raw_path).name)
    finger_order = [f"{side}.{name}" for side in SIDES for name in links[side]]
    max_step = 0; previous_obs_index = -1; previous_anchor = -1
    used = deque(maxlen=window_steps)
    pending = next(raw_iter, None)
    tick_span = window_steps // bins
    for marker in obs_iter:
        if not isinstance(marker, dict):
            raise ValueError("invalid observation marker")
        obs_index, anchor = int(marker["obs_index"]), int(marker["physics_step"])
        if type(marker.get("valid")) is not bool:
            raise ValueError("observation marker valid must be a boolean")
        if obs_index != previous_obs_index + 1 or anchor < previous_anchor or anchor < 0:
            raise ValueError("observation markers must have consecutive indices and nondecreasing physics steps")
        previous_obs_index, previous_anchor = obs_index, anchor
        if not math.isclose(float(marker["sim_time_s"]), round(anchor * dt, 8), abs_tol=2e-7):
            raise ValueError("observation marker time does not match physics step")
        while pending is not None and int(pending["physics_step"]) <= anchor:
            step = int(pending["physics_step"])
            if step <= max_step or step <= 0:
                raise ValueError("nonmonotonic or duplicate physics step")
            if not math.isclose(float(pending["sim_time_s"]), round(step * dt, 8), abs_tol=2e-7):
                raise ValueError("physics sample time does not match physics step")
            if not isinstance(pending.get("left"), dict) or not isinstance(pending.get("right"), dict):
                raise ValueError("missing left/right force sample")
            max_step = step
            used.append(pending)
            pending = next(raw_iter, None)
        if marker.get("valid", False) and (not used or int(used[-1]["physics_step"]) != anchor):
            raise ValueError("marked valid but no same-step force sample")
        by_step = {int(row["physics_step"]): row for row in used if int(row["physics_step"]) > anchor-window_steps}
        shape = [bins, len(finger_order), len(FEATURES)]
        matrix = []; masks = []; coverage = []; observed_max = 0.0
        per_finger_complete = True
        for bin_i in range(bins):
            begin = anchor-window_steps+1 + tick_span*bin_i
            steps = [by_step.get(tick) for tick in range(begin, begin+tick_span)]
            valid_steps = [row for row in steps if row is not None]
            coverage.append(len(valid_steps)/tick_span)
            bin_values, bin_mask = [], []
            for side in SIDES:
                for finger in links[side]:
                    values = []
                    for row in valid_steps:
                        fingers = row[side].get("fingers", {})
                        if finger not in fingers:
                            per_finger_complete = False
                            continue
                        value = float(fingers[finger]["load_proxy_n"])
                        if value < 0 or not math.isfinite(value):
                            raise ValueError("invalid nonnegative finger load")
                        values.append(value)
                    # Mean is over observed ticks, with mask distinguishing missing ticks.
                    observed_max = max(observed_max, max(values, default=0.0))
                    if values:
                        bin_values.append([sum(values)/len(values), max(values),
                                           sum(value > 0 for value in values)/len(values)])
                    else:
                        bin_values.append([0.0, 0.0, 0.0])
                    bin_mask.append(len(values)/tick_span)
            matrix.append(bin_values)
            masks.append(bin_mask)
        observed = sorted(by_step)
        last_source = observed[-1] if observed else None
        age_ms = (anchor-last_source)*dt*1000 if last_source is not None else None
        complete = (anchor >= window_steps and len(observed) == window_steps and
                    observed[0] == anchor-window_steps+1 and observed[-1] == anchor and
                    per_finger_complete)
        high_impact = observed_max > peak_warning_n
        yield {"schema": "phizju_tactile_window_v1", "task": raw_head["task"],
               "seed": raw_head["seed"], "episode": raw_head["episode"],
               "observation_marker_index": obs_index, "anchor_physics_step": anchor,
               "anchor_sim_time_s": round(anchor*dt, 8), "camera_shapes": marker.get("rgb_shapes"),
               "marker_role": "get_obs_capture_not_necessarily_policy_inference",
               "source_force_file": Path(raw_path).name,
               "window_steps": window_steps, "dt_s": dt, "shape": shape,
               "finger_order": finger_order, "feature_order": list(FEATURES),
               "features": matrix, "sample_fraction_mask": masks,
               "quality": {"coverage": len(observed)/window_steps,
                           "bins_coverage": coverage, "last_source_step": last_source,
                           "force_age_ms": age_ms, "per_finger_complete": per_finger_complete,
                           "high_impact_warning": high_impact,
                           "window_valid": complete and bool(marker.get("valid", False))}}
    # Validate remaining raw records without ever letting them enter prior windows.
    while pending is not None:
        step = int(pending["physics_step"])
        if step <= max_step or step <= 0:
            raise ValueError("nonmonotonic or duplicate physics step")
        if not math.isclose(float(pending["sim_time_s"]), round(step * dt, 8), abs_tol=2e-7):
            raise ValueError("physics sample time does not match physics step")
        max_step = step
        pending = next(raw_iter, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--markers", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window-steps", type=int, default=25)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--peak-warning-n", type=float, default=500)
    args = parser.parse_args()
    if args.out.resolve() in {args.raw.resolve(), args.markers.resolve()}:
        parser.error("output must differ from inputs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from tempfile import NamedTemporaryFile
    # Do not leave a partial dataset on validation failure.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.out.parent,
                            prefix=".tactile_windows_", delete=False) as file:
        temporary = Path(file.name)
        try:
            count = ready = 0
            for item in align(args.raw, args.markers, window_steps=args.window_steps,
                              bins=args.bins, peak_warning_n=args.peak_warning_n):
                file.write(json.dumps(item, allow_nan=False) + "\n")
                count += 1; ready += item["quality"]["window_valid"]
        except Exception:
            file.close(); temporary.unlink(missing_ok=True)
            raise
    temporary.replace(args.out)
    print(f"observation candidates={count}; complete force windows={ready}; output={args.out}")


if __name__ == "__main__":
    main()

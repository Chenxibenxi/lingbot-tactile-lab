"""Read-only, CPU-only physical-clock feasibility audit of *saved-frame* 14D targets.

H is a number of future saved frames, not a number of physics-step actions.
Neither the split's test fold nor future force/images enter this audit. Results
are exploratory candidates; they do not approve fitting a LingBot action chunk.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from build_paired_index import build as paired_index
from register_dataset import build_card
from validate_expert_data import sha256

HORIZONS = (16, 50)
TASK_SPLITS = {
    "lift_pot": "relay_lift_pot_seed_split_candidate_v1.json",
    "pick_diverse_bottles": "relay_pickdiverse_seed_split_candidate_v2.json",
}
FOLDS = ("train", "validation")
STATUS = "candidate_not_training_approved"
ROLE = "relay_single_rtx3090"


def select_chunks(rows, horizon, *, saved_frames, physics_steps, dt_s, trial_id):
    """Pure candidate selection from one validated paired-index episode.

    ``rows`` includes *all* t -> t+1 pairs (including invalid tactile anchors).
    The final saved frame appears as the last row's target, never as a new
    anchor. Any nonpositive gap invalidates every window crossing that edge.
    """
    if horizon not in HORIZONS or not isinstance(saved_frames, int) or saved_frames < 2:
        raise ValueError("unsupported horizon or saved frame count")
    if not isinstance(physics_steps, int) or physics_steps < 0 or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("invalid physical clock")
    rows = list(rows)
    if len(rows) != saved_frames - 1:
        raise ValueError("paired rows must cover every saved-frame transition")
    frames = []
    identity = None
    evidence = None
    for i, row in enumerate(rows):
        current = (row.get("task"), row.get("seed"), row.get("episode"))
        if any(value is None for value in current) or (identity is not None and current != identity):
            raise ValueError("cross-episode paired rows")
        identity = current
        inp, target = row["input"], row["target"]
        step = inp["capture_physics_step"]
        if (row.get("sample_index") != i or inp.get("frame_index") != i or
                target.get("action_source_frame_index") != i + 1 or
                not isinstance(step, int) or step < 0 or step > physics_steps or
                not isinstance(target.get("next_frame_physics_step"), int) or
                target["next_frame_physics_step"] > physics_steps):
            raise ValueError("frame index or physical step outside same episode")
        state = inp["command_state_14"]
        if len(state) != 14 or any(not math.isfinite(float(x)) for x in state):
            raise ValueError("invalid 14D saved-frame state")
        this_evidence = row.get("evidence", {})
        if (not isinstance(this_evidence.get("hdf5_sha256"), str) or
                not isinstance(this_evidence.get("raw_sha256"), str) or
                (evidence is not None and this_evidence != evidence)):
            raise ValueError("missing or inconsistent source SHA")
        evidence = this_evidence
        if i and (rows[i - 1]["target"]["next_frame_physics_step"] != step or
                  rows[i - 1]["target"]["action_command_14"] != state):
            raise ValueError("future target differs from same-HDF5 next saved frame")
        frames.append((step, list(state)))
    last = rows[-1]["target"]
    if (last["next_frame_physics_step"] < 0 or
            len(last["action_command_14"]) != 14 or
            any(not math.isfinite(float(x)) for x in last["action_command_14"])):
        raise ValueError("invalid final saved-frame target")
    frames.append((last["next_frame_physics_step"], list(last["action_command_14"])))
    results = []
    stats = {"eligible_anchors": 0, "legal_windows": 0, "all_15_step_windows": 0,
             "rejected_nonpositive_gap": 0, "rejected_out_of_bounds": 0}
    for t, row in enumerate(rows):
        if row["input"].get("tactile_window_valid") is not True:
            continue
        stats["eligible_anchors"] += 1
        if t + horizon >= saved_frames:
            stats["rejected_out_of_bounds"] += 1
            continue
        steps = [frame[0] for frame in frames[t:t + horizon + 1]]
        gaps = [b - a for a, b in zip(steps, steps[1:])]
        if any(gap <= 0 for gap in gaps):
            stats["rejected_nonpositive_gap"] += 1
            continue
        offsets = [{"offset": k, "frame_index": t + k,
                    "physics_step": steps[k], "delta_physics_steps": steps[k] - steps[0],
                    "gap_from_previous_physics_steps": gaps[k - 1],
                    "drive_target_state_14": frames[t + k][1]}
                   for k in range(1, horizon + 1)]
        all_15 = all(gap == 15 for gap in gaps)
        results.append({"schema": "lingbot_physical_saved_frame_chunk_candidate_v1",
                        "status": STATUS, "trial_id": trial_id, "task": identity[0],
                        "seed": identity[1], "episode": identity[2], "anchor_frame_index": t,
                        "anchor_physics_step": steps[0], "tactile_window_valid": True,
                        "horizon_saved_frames": horizon, "physics_dt_s": dt_s,
                        "last_target_elapsed_s": (steps[-1] - steps[0]) * dt_s,
                        "all_adjacent_gaps_15_physics_steps": all_15,
                        "target_semantics": "future_saved_frame_14D_drive_target_not_physics_step_actions",
                        "hdf5_sha256": evidence["hdf5_sha256"],
                        "frames_sha256": evidence.get("frames_sha256"),
                        "targets": offsets})
        stats["legal_windows"] += 1
        stats["all_15_step_windows"] += int(all_15)
    return results, stats


def _split(repo, task, filename):
    path = repo / "datasets" / filename
    split = json.loads(path.read_text(encoding="utf-8"))
    if (split.get("schema") != "lingbot_tactile_seed_split_plan_v1" or
            split.get("status") != "candidate_not_approved" or
            split.get("source_machine_role") != ROLE or
            split.get("evaluation_claims_approved") is not False or
            split.get("random_seed") != 42):
        raise ValueError("split provenance or original seed changed")
    groups = split.get("task_seed_groups", {}).get(task)
    if not isinstance(groups, dict):
        raise ValueError("missing task seed registry")
    ids = [trial for fold in (*FOLDS, "test") for trial in split.get(fold, [])]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_]+", str(i)) for i in ids):
        raise ValueError("duplicate or malformed trial IDs across folds")
    seeds = [seed for fold in (*FOLDS, "test") for seed in groups.get(fold, [])]
    if len(seeds) != len(set(seeds)):
        raise ValueError("actual seed crosses split folds")
    if task == "pick_diverse_bottles":
        original_path = repo / "datasets/relay_pickdiverse_seed_split_candidate_v1.json"
        original = json.loads(original_path.read_text(encoding="utf-8"))
        if (split.get("parent_v1_split_sha256") != sha256(original_path) or
                any(split[fold] != original[fold] or groups[fold] != original["task_seed_groups"][task][fold]
                    for fold in ("validation", "test")) or
                not set(original["train"]).issubset(split["train"]) or
                not set(original["task_seed_groups"][task]["train"]).issubset(groups["train"]) or
                any(split["expert_card_sha256"].get(i) != original["expert_card_sha256"].get(i)
                    for i in original["train"] + original["validation"] + original["test"])):
            raise ValueError("bottle v2 changed old split, heldout seed or source SHA")
    return split, sha256(path)


def build(repo):
    """Revalidate relay cards/HDF5/frame SHA; inspect train+validation ONLY."""
    repo = Path(repo).resolve(strict=True)
    registry_path = repo / "datasets/provenance_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema") != "lingbot_tactile_dataset_registry_v1" or registry.get("approved_paired_episodes") != []:
        raise ValueError("only unapproved candidate provenance is supported")
    records = []
    summary = {"schema": "lingbot_physical_action_chunk_audit_v1", "status": STATUS,
               "source_machine_role": ROLE, "horizons_saved_frames": list(HORIZONS),
               "target_semantics": "future_saved_frame_14D_drive_target_not_physics_step_actions",
               "physical_clock": "same_episode_physics_step_and_dt_not_lerobot_50fps",
               "training_approval": False, "test_future_labels_read": False,
               "provenance_registry_sha256": sha256(registry_path), "tasks": {}}
    for task, name in TASK_SPLITS.items():
        split, split_sha = _split(repo, task, name)
        folds = {}
        for fold in FOLDS:
            trial_ids = split[fold]
            expected_seeds = split["task_seed_groups"][task][fold]
            if not trial_ids or len(trial_ids) != len(expected_seeds):
                raise ValueError("missing or extra trial/seed in fold")
            counters = {h: {"eligible_anchors": 0, "legal_windows": 0,
                            "all_15_step_windows": 0, "rejected_nonpositive_gap": 0,
                            "rejected_out_of_bounds": 0} for h in HORIZONS}
            actual_seeds = set()
            for trial_id in trial_ids:
                card_path = repo / "datasets/cards" / f"{trial_id}.json"
                card = json.loads(card_path.read_text(encoding="utf-8"))
                trial = repo / "data/paired" / trial_id
                if ("_expert" not in trial_id or sha256(card_path) != split["expert_card_sha256"].get(trial_id) or
                        card != build_card(trial, repo) or card["source_machine_role"] != ROLE or
                        card["task"] != task or card["trial_id"] != trial_id or
                        card["seed"] not in expected_seeds or card["seed"] in actual_seeds):
                    raise ValueError("relay expert card, source SHA, task or fold failed validation")
                actual_seeds.add(card["seed"])
                raw_files = [p for p in (trial / "force").glob("*.jsonl")
                             if not p.name.endswith((".obs.jsonl", ".frames.jsonl", ".policy.jsonl"))]
                if len(raw_files) != 1:
                    raise ValueError("expected one expert force stream")
                raw = raw_files[0]
                hdf5 = trial / "demonstration/data" / f"episode{card['episode']}.hdf5"
                rows = list(paired_index(raw, raw.with_suffix(".obs.jsonl"),
                                         raw.with_suffix(".frames.jsonl"), hdf5,
                                         trial / "completion.json"))
                if any((r["task"], r["seed"], r["episode"]) !=
                       (task, card["seed"], card["episode"]) or
                       r["evidence"]["hdf5_sha256"] != card["sha256"]["hdf5"] or
                       r["evidence"]["raw_sha256"] != card["sha256"]["raw"] for r in rows):
                    raise ValueError("paired row SHA or episode identity changed")
                for h in HORIZONS:
                    selected, stats = select_chunks(rows, h, saved_frames=card["saved_frames"],
                        physics_steps=card["physics_steps"], dt_s=card["capture_dt_s"], trial_id=trial_id)
                    for key, count in stats.items():
                        counters[h][key] += count
                    for row in selected:
                        row["fold"] = fold
                        row["frames_sha256"] = card["sha256"]["frames"]
                        row["expert_card_sha256"] = split["expert_card_sha256"][trial_id]
                    records.extend(selected)
            if actual_seeds != set(expected_seeds):
                raise ValueError("fold's actual expert seeds differ from original registry")
            folds[fold] = {"trials": len(trial_ids), "horizons": {str(h): counters[h] for h in HORIZONS}}
        summary["tasks"][task] = {"split_sha256": split_sha, "folds": folds}
    pot = summary["tasks"]["lift_pot"]["folds"]
    gate = all(pot[fold]["horizons"]["50"]["all_15_step_windows"] > 0 for fold in FOLDS)
    summary["pot_h50_train_validation_all15_gate_pass"] = gate
    summary["original_lingbot_h50_training_feasible"] = False  # audit alone cannot approve training
    summary["gate_reason"] = ("candidate window counts exist but approval requires independent review" if gate else
                              "FAIL_CLOSED: pot H50 has zero genuine all-15-step train or validation windows")
    summary["private_candidate_rows"] = len(records)
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--private-out", type=Path, help="optional ignored data/reports candidate JSONL; exclusive create")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    out = None
    if args.private_out is not None:
        out = args.private_out.resolve()
        reports = (repo / "data/reports").resolve(strict=True)
        if out.parent != reports or out.exists() or not out.name.endswith(".jsonl"):
            parser.error("new .jsonl output must be directly in private data/reports; overwrite forbidden")
        import subprocess
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(out)],
                                 cwd=repo, check=False, capture_output=True)
        if ignored.returncode != 0:
            parser.error("private output path must be Git ignored")
    records, report = build(repo)
    if out is not None:
        with out.open("x", encoding="utf-8") as file:
            for row in records:
                file.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

"""Create a provenance card only for validated *local relay-3090* paired trials.

No bytes move from the trial directory. Cards are candidates, not automatic
training approval; a split manifest and quality review are still required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from validate_expert_data import sha256, validate

ROLE = "relay_single_rtx3090"
HOST = "admin001-MS-7D90"


def build_card(trial: Path, repo: Path) -> dict:
    repo = repo.resolve(strict=True)
    trial = trial.resolve(strict=True)
    paired_root = (repo / "data/paired").resolve(strict=True)
    if trial.parent != paired_root:
        raise ValueError("trial must live directly under this repo's relay data/paired")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", trial.name):
        raise ValueError("invalid trial ID")
    completion = trial / "completion.json"
    meta = json.loads(completion.read_text())
    if (meta.get("source_machine_role"), meta.get("source_hostname")) != (ROLE, HOST):
        raise ValueError("non-relay provenance: refuse external-machine or unverified data")
    if meta.get("schema") != "phizju_expert_episode_complete_v1" or not meta.get("success"):
        raise ValueError("trial lacks a successful completion marker")
    raw_files = [p for p in (trial / "force").glob("*.jsonl")
                 if not p.name.endswith((".obs.jsonl", ".frames.jsonl"))]
    if len(raw_files) != 1:
        raise ValueError("expected exactly one raw physics stream")
    raw = raw_files[0]
    hdf5 = trial / "demonstration/data" / f"episode{meta['episode']}.hdf5"
    result = validate(raw, raw.with_suffix(".obs.jsonl"),
                      raw.with_suffix(".frames.jsonl"), hdf5, completion)
    return {"schema": "lingbot_tactile_dataset_card_v1", "status": "candidate_unassigned_split",
            "trial_id": trial.name, "source_machine_role": ROLE, "source_hostname": HOST,
            "task": result["task"], "seed": result["seed"], "episode": result["episode"],
            "physics_steps": result["physics_steps"], "saved_frames": result["saved_frames"],
            "capture_dt_s": result["dt_s"], "sha256": result["sha256"],
            "completion_sha256": sha256(completion),
            "data_path_relative": str(trial.relative_to(repo)),
            "image_hdf5_relative": str(hdf5.relative_to(repo)),
            "split": None, "calibrated_grip_force": False,
            "note": "sim impulse/dt proxy, not GelSight and not a real hardware force limit"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--out", type=Path, help="candidate card output (default datasets/cards/<trial>.json)")
    args = p.parse_args()
    card = build_card(args.trial, args.repo)
    out = args.out or args.repo / "datasets/cards" / (card["trial_id"] + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as file:
        json.dump(card, file, indent=2)
        file.write("\n")
    print("CANDIDATE_REGISTERED", out, "frames", card["saved_frames"])


if __name__ == "__main__":
    main()

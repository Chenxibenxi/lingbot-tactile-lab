"""Synthetic CPU-only checks; no relay data, GPU, model or test-fold labels."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physical_action_chunks import _split, build, select_chunks


def example_rows(steps, *, episode=0, valid=None):
    valid = set(range(len(steps))) if valid is None else valid
    rows = []
    for t in range(len(steps) - 1):
        rows.append({"task": "lift_pot", "seed": 123, "episode": episode,
                     "sample_index": t,
                     "input": {"frame_index": t, "capture_physics_step": steps[t],
                               "command_state_14": [float(t)] * 14,
                               "tactile_window_valid": t in valid},
                     "target": {"action_source_frame_index": t + 1,
                                "next_frame_physics_step": steps[t + 1],
                                "action_command_14": [float(t + 1)] * 14},
                     "evidence": {"hdf5_sha256": "a" * 64, "raw_sha256": "b" * 64}})
    return rows


def select(rows, h=16):
    return select_chunks(rows, h, saved_frames=len(rows) + 1,
                         physics_steps=1000, dt_s=.004, trial_id="relay_expert0")


class PhysicalChunksTest(unittest.TestCase):
    def test_all_15_positive_sixteen_targets_are_saved_frames_not_ticks(self):
        rows = example_rows(list(range(0, 20 * 15, 15)), valid={1, 2})
        found, stats = select(rows)
        self.assertEqual(stats["all_15_step_windows"], 2)
        self.assertEqual([r["anchor_frame_index"] for r in found], [1, 2])
        self.assertEqual(found[0]["targets"][0]["frame_index"], 2)
        self.assertEqual(found[0]["targets"][0]["physics_step"], 30)
        self.assertEqual(found[0]["targets"][-1]["delta_physics_steps"], 240)
        self.assertAlmostEqual(found[0]["last_target_elapsed_s"], .96)
        self.assertEqual(len(found[0]["targets"][-1]["drive_target_state_14"]), 14)
        self.assertNotIn("future_force", str(found))
        self.assertEqual(found[0]["status"], "candidate_not_training_approved")

    def test_duplicate_and_nonpositive_gap_rejected_without_fake_fps(self):
        for bad_step in (15, 9):
            steps = list(range(0, 19 * 15, 15))
            steps[2] = bad_step
            selected, stats = select(example_rows(steps, valid={0}))
            self.assertFalse(selected)
            self.assertEqual(stats["rejected_nonpositive_gap"], 1)

    def test_irregular_positive_gaps_count_legal_not_all15(self):
        steps = list(range(0, 19 * 15, 15))
        steps[4] += 2
        selected, stats = select(example_rows(steps, valid={0}))
        self.assertEqual(len(selected), 1)
        self.assertEqual(stats["all_15_step_windows"], 0)
        self.assertEqual(selected[0]["targets"][3]["gap_from_previous_physics_steps"], 17)
        self.assertAlmostEqual(selected[0]["last_target_elapsed_s"], .96)
        self.assertEqual(selected[0]["targets"][4]["gap_from_previous_physics_steps"], 13)

    def test_future_out_of_bounds_and_invalid_anchor(self):
        found, stats = select(example_rows(list(range(0, 17 * 15, 15)), valid={1, 15}))
        self.assertEqual(found, [])
        self.assertEqual(stats["eligible_anchors"], 2)
        self.assertEqual(stats["rejected_out_of_bounds"], 2)

    def test_cross_episode_forged_target_and_invalid_frame_rejected(self):
        rows = example_rows(list(range(0, 19 * 15, 15)))
        rows[3]["episode"] = 1
        with self.assertRaisesRegex(ValueError, "cross-episode"):
            select(rows)
        rows = example_rows(list(range(0, 19 * 15, 15)))
        rows[0]["target"]["action_command_14"] = [99.0] * 14
        with self.assertRaisesRegex(ValueError, "same-HDF5"):
            select(rows)
        rows = example_rows(list(range(0, 19 * 15, 15)))
        rows[1]["input"]["frame_index"] = 99
        with self.assertRaisesRegex(ValueError, "frame index"):
            select(rows)

    def test_h50_requires_51_saved_frames(self):
        rows = example_rows(list(range(0, 51 * 15, 15)), valid={0})
        found, stats = select(rows, h=50)
        self.assertEqual((len(found), stats["all_15_step_windows"]), (1, 1))
        self.assertEqual(found[0]["targets"][-1]["frame_index"], 50)
        self.assertAlmostEqual(found[0]["last_target_elapsed_s"], 3.0)

    def test_build_fail_closed_with_no_h50_pot_and_never_reads_test_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "datasets/cards").mkdir(parents=True)
            (repo / "data/paired").mkdir(parents=True)
            (repo / "datasets/provenance_v1.json").write_text(json.dumps({
                "schema": "lingbot_tactile_dataset_registry_v1", "approved_paired_episodes": []}))
            splits = {"schema": "lingbot_tactile_seed_split_plan_v1", "status": "candidate_not_approved",
                      "source_machine_role": "relay_single_rtx3090", "evaluation_claims_approved": False,
                      "random_seed": 42, "train": ["train_expert0"], "validation": ["val_expert0"],
                      "test": ["test_expert0"], "task_seed_groups": {"lift_pot": {
                          "train": [1], "validation": [2], "test": [3]}},
                      "expert_card_sha256": {key: "c" * 64 for key in
                                             ("train_expert0", "val_expert0", "test_expert0")}}
            (repo / "datasets/pot.json").write_text(json.dumps(splits))
            cards = {}
            for trial, seed in (("train_expert0", 1), ("val_expert0", 2)):
                directory = repo / "data/paired" / trial
                (directory / "force").mkdir(parents=True)
                (directory / "force/run.jsonl").touch()
                card = {"trial_id": trial, "task": "lift_pot", "seed": seed, "episode": 0,
                        "source_machine_role": "relay_single_rtx3090", "saved_frames": 20,
                        "physics_steps": 400, "capture_dt_s": .004,
                        "sha256": {"raw": "b" * 64, "hdf5": "a" * 64,
                                   "frames": "d" * 64}}
                cards[trial] = card
                (repo / "datasets/cards" / (trial + ".json")).write_text(json.dumps(card))
            def fake_hash(path):
                return "c" * 64
            def fake_rows(raw, obs, frames, hdf5, completion):
                self.assertNotIn("test_expert0", str(raw))
                rows = example_rows(list(range(0, 20 * 15, 15)), valid={1})
                for row in rows:
                    row["seed"] = 1 if "train_expert0" in str(raw) else 2
                return rows
            with patch("physical_action_chunks.TASK_SPLITS", {"lift_pot": "pot.json"}), \
                 patch("physical_action_chunks.sha256", side_effect=fake_hash), \
                 patch("physical_action_chunks.build_card", side_effect=lambda trial, root: cards[trial.name]), \
                 patch("physical_action_chunks.paired_index", side_effect=fake_rows):
                rows, report = build(repo)
            self.assertEqual(len(rows), 2)
            self.assertEqual(report["tasks"]["lift_pot"]["folds"]["train"]["horizons"]["16"]["all_15_step_windows"], 1)
            self.assertFalse(report["pot_h50_train_validation_all15_gate_pass"])
            self.assertFalse(report["original_lingbot_h50_training_feasible"])
            self.assertFalse(report["test_future_labels_read"])
            self.assertIn("FAIL_CLOSED", report["gate_reason"])

    def test_split_rejects_cross_fold_seeds_without_reading_test_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "datasets").mkdir()
            path = repo / "datasets/relay_lift_pot_seed_split_candidate_v1.json"
            split = {"schema": "lingbot_tactile_seed_split_plan_v1", "status": "candidate_not_approved",
                     "source_machine_role": "relay_single_rtx3090", "evaluation_claims_approved": False,
                     "random_seed": 42, "train": ["train_expert0"],
                     "validation": ["val_expert0"], "test": ["heldout_expert0"],
                     "task_seed_groups": {"lift_pot": {"train": [1], "validation": [2], "test": [3]}}}
            path.write_text(json.dumps(split))
            self.assertEqual(_split(repo, "lift_pot", path.name)[0]["test"], ["heldout_expert0"])
            split["task_seed_groups"]["lift_pot"]["test"] = [1]
            path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, "crosses"):
                _split(repo, "lift_pot", path.name)


if __name__ == "__main__":
    unittest.main()

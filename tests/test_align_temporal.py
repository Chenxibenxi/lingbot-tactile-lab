import json
import tempfile
import unittest
from pathlib import Path

from align_temporal import align


def fixture(root, *, missing=(), legacy=False, seed=7):
    raw, obs = root / "trial.jsonl", root / "trial.obs.jsonl"
    h = {"schema": "phizju_force_probe_v1", "task": "lift_pot", "seed": seed,
         "episode": 0, "dt_s": .004, "side_links": {
             "left": ["fl_link7", "fl_link8"], "right": ["fr_link7", "fr_link8"]}}
    rows = [h]
    for i in range(1, 31):
        if i in missing:
            continue
        def side(prefix):
            fingers = {f"{prefix}_link{j}": {"load_proxy_n": float(i) if j == 7 else 0.0,
                                               "points": 1 if j == 7 else 0}
                       for j in (7, 8)}
            result = {"load_proxy_n": float(i), "points": 1}
            if not legacy:
                result["fingers"] = fingers
            return result
        rows.append({"physics_step": i, "sim_time_s": round(i * .004, 8),
                     "action_index": 1, "left": side("fl"), "right": side("fr")})
    raw.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    events = [{"schema": "phizju_obs_marker_v1", "probe_file": raw.name,
               "task": "lift_pot", "seed": 7, "episode": 0, "dt_s": .004}]
    for index, step in enumerate((0, 25, 30)):
        events.append({"obs_index": index, "physics_step": step,
                       "sim_time_s": round(step*.004, 8), "action_index": index,
                       "valid": step != 0, "rgb_shapes": {"head_camera": [240, 320, 3]}})
    obs.write_text("\n".join(json.dumps(x) for x in events)+"\n")
    return raw, obs


class AlignTest(unittest.TestCase):
    def test_fixed_shape_causal_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, obs = fixture(Path(temp))
            first, middle, last = list(align(raw, obs))
            self.assertEqual(first["anchor_physics_step"], 0)
            self.assertFalse(first["quality"]["window_valid"])
            self.assertEqual(middle["shape"], [5, 4, 3])
            self.assertEqual(middle["quality"]["coverage"], 1)
            self.assertEqual(middle["quality"]["last_source_step"], 25)
            self.assertTrue(middle["quality"]["window_valid"])
            self.assertEqual(middle["features"][0][0], [3.0, 5.0, 1.0])
            self.assertEqual(middle["features"][0][1], [0.0, 0.0, 0.0])
            self.assertEqual(last["quality"]["last_source_step"], 30)
            self.assertEqual(last["features"][-1][0], [28.0, 30.0, 1.0])

    def test_future_peak_not_visible_at_earlier_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, obs = fixture(Path(temp))
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            rows[-1]["left"]["fingers"]["fl_link7"]["load_proxy_n"] = 999.0
            raw.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
            early, late = list(align(raw, obs))[1:]
            self.assertLess(max(bin_[0][1] for bin_ in early["features"]), 100)
            self.assertEqual(late["features"][-1][0][1], 999.0)
            self.assertTrue(late["quality"]["high_impact_warning"])
            self.assertTrue(late["quality"]["window_valid"])  # warning != silent exclusion

    def test_missing_tick_is_masked_not_imputed(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, obs = fixture(Path(temp), missing=(20,))
            middle = list(align(raw, obs))[1]
            self.assertAlmostEqual(middle["quality"]["coverage"], 24/25)
            self.assertFalse(middle["quality"]["window_valid"])
            self.assertEqual(middle["sample_fraction_mask"][3][0], .8)
            self.assertEqual(middle["features"][3][0][0], (16+17+18+19)/4)

    def test_legacy_without_per_finger_never_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, obs = fixture(Path(temp), legacy=True)
            middle = list(align(raw, obs))[1]
            self.assertFalse(middle["quality"]["per_finger_complete"])
            self.assertFalse(middle["quality"]["window_valid"])
            self.assertEqual(middle["sample_fraction_mask"][0], [0.0]*4)

    def test_mismatched_episode_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, obs = fixture(Path(temp), seed=8)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                list(align(raw, obs))

    def test_missing_current_force_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, obs = fixture(Path(temp), missing=(25,))
            with self.assertRaisesRegex(ValueError, "same-step"):
                list(align(raw, obs))

    def test_duplicate_raw_step_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, obs = fixture(Path(temp))
            rows = raw.read_text().splitlines()
            rows.insert(4, rows[3])
            raw.write_text("\n".join(rows)+"\n")
            with self.assertRaisesRegex(ValueError, "duplicate physics step"):
                list(align(raw, obs))

    def test_marker_order_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, obs = fixture(Path(temp))
            rows = [json.loads(x) for x in obs.read_text().splitlines()]
            rows[-1]["obs_index"] = 9
            obs.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
            with self.assertRaisesRegex(ValueError, "consecutive"):
                list(align(raw, obs))


if __name__ == "__main__":
    unittest.main()

"""No SAPIEN dependency: verify sign, units, near contacts and opt-in patch anchors."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from force_probe import ForceProbe, gripper_links, sample_contacts
from make_overlay import patched_source


def body(name):
    return NS(entity=NS(name=name))


def point(impulse):
    return NS(impulse=impulse)


def contact(a, b, *values):
    return NS(bodies=[body(a), body(b)], points=[point(v) for v in values])


def fake_robot():
    def joint(name):
        return NS(child_link=NS(get_name=lambda: name))
    return NS(left_gripper=[(joint("fl_link7"), 1, 0), (joint("fl_link8"), -1, 0)],
              right_gripper=[(joint("fr_link7"), 1, 0), (joint("fr_link8"), -1, 0)],
              left_fix_gripper_name=[], right_fix_gripper_name=[])


class ForceProbeTest(unittest.TestCase):
    def test_contact_impulse_sign_and_units(self):
        scene = NS(get_contacts=lambda: [
            contact("fl_link7", "cube", [0.02, 0, 0]),
            contact("cube", "fr_link8", [0.01, 0, 0]),
            contact("fl_link7", "fl_link8", [50, 0, 0]),  # finger-finger excluded
            contact("fl_link7", "left_camera", [50, 0, 0]),  # other robot link excluded
            contact("fr_link7", "table", [0, 0, 0]),  # proximity, not touch
        ])
        result = sample_contacts(scene, gripper_links(fake_robot()), .004,
                                 robot_links={"fl_link7", "fl_link8", "left_camera"})
        self.assertEqual(result["left"]["contacts"], 1)
        self.assertEqual(result["left"]["points"], 1)
        self.assertAlmostEqual(result["left"]["load_proxy_n"], 5)
        self.assertEqual(result["left"]["objects"], ["cube"])
        self.assertAlmostEqual(result["left"]["fingers"]["fl_link7"]["load_proxy_n"], 5)
        self.assertEqual(result["left"]["fingers"]["fl_link8"]["points"], 0)
        self.assertAlmostEqual(result["right"]["net_impulse_world_ns"][0], -.01)
        self.assertAlmostEqual(result["right"]["load_proxy_n"], 2.5)
        self.assertEqual(result["right"]["objects"], ["cube"])

    def test_missing_impulse_fails_closed(self):
        scene = NS(get_contacts=lambda: [NS(bodies=[body("cube"), body("fl_link7")],
                                             points=[NS(position=[0, 0, 0])])])
        with self.assertRaises(RuntimeError):
            sample_contacts(scene, gripper_links(fake_robot()), .004)
        with self.assertRaises(ValueError):
            sample_contacts(scene, gripper_links(fake_robot()), 0)

    def test_probe_does_not_change_scene_and_writes_jsonl(self):
        class Scene:
            calls = 0
            def get_contacts(self):
                self.calls += 1
                return [contact("fl_link8", "cube", [0, 0, .004])]
        with tempfile.TemporaryDirectory() as td:
            scene = Scene(); robot = fake_robot()
            probe = ForceProbe(td, task="lift_pot", seed=7, episode=2, dt=.004, robot=robot)
            probe.sample(scene, robot, 1); probe.sample(scene, robot, 1); probe.close()
            rows = [json.loads(x) for x in next(Path(td).glob("*.jsonl")).read_text().splitlines()]
            self.assertEqual(scene.calls, 2)
            self.assertEqual(rows[0]["schema"], "phizju_force_probe_v1")
            self.assertEqual(rows[0]["side_links"]["left"], ["fl_link7", "fl_link8"])
            self.assertEqual(rows[1]["physics_step"], 1)
            self.assertAlmostEqual(rows[2]["left"]["load_proxy_n"], 1)
            self.assertTrue(probe.observation()["left_contact"])
            self.assertAlmostEqual(probe.observation()["left_load_proxy_n"], 1)

    def test_observation_markers_pair_with_raw_steps(self):
        with tempfile.TemporaryDirectory() as td:
            robot = fake_robot()
            probe = ForceProbe(td, task="lift_pot", seed=7, episode=2, dt=.004, robot=robot)
            frame = {"observation": {"head_camera": {"rgb": NS(shape=(240, 320, 3))}}}
            probe.record_observation(frame, 0)
            scene = NS(get_contacts=lambda: [])
            probe.sample(scene, robot, 1)
            probe.record_observation(frame, 1)
            probe.close()
            raw_files = [p for p in Path(td).glob("*.jsonl") if not p.name.endswith(".obs.jsonl")]
            self.assertEqual(len(raw_files), 1)
            rows = [json.loads(x) for x in raw_files[0].with_suffix(".obs.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["probe_file"], raw_files[0].name)
            self.assertEqual([(r["obs_index"], r["physics_step"], r["action_index"])
                              for r in rows[1:]], [(0, 0, 0), (1, 1, 1)])
            self.assertEqual(rows[1]["rgb_shapes"]["head_camera"], [240, 320, 3])
            self.assertFalse(rows[1]["valid"])
            self.assertTrue(rows[2]["valid"])

    def test_duplicate_links_rejected(self):
        robot = fake_robot(); robot.right_fix_gripper_name = ["fl_link7"]
        with self.assertRaises(ValueError):
            gripper_links(robot)

    def test_isolated_patch_preserves_checks(self):
        source = (Path(os.environ.get("ROBOTWIN_SOURCE", "/mnt/relaydata/lingbot/RoboTwin")) /
                  "envs/_base_task.py").read_text()
        patched = patched_source(source)
        self.assertIn("self._force_probe.sample(self.scene, self.robot, self.take_action_cnt)", patched)
        self.assertIn('pkl_dic["tactile_force"] = self._force_probe.observation()', patched)
        self.assertIn('self._force_probe.record_observation(pkl_dic, self.take_action_cnt)', patched)
        self.assertEqual(source.count("if self.check_success():"),
                         patched.count("if self.check_success():"))
        self.assertEqual(source.count("self.scene.step()"), patched.count("self.scene.step()"))


if __name__ == "__main__":
    unittest.main()

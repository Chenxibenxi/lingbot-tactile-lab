"""One isolated RoboTwin scene/action: verify opt-in force field reaches get_obs.

No policy server, weight loads, success scoring, teleoperation or training.
Run only inside an overlay made by make_overlay.py and with idle sim GPU.
"""
import json
import os
from pathlib import Path


def main():
    import numpy as np
    import sapien
    import yaml
    from envs.lift_pot import lift_pot
    root = Path.cwd()
    if not (root / "envs/_base_task.py").is_file() or not os.environ.get("FORCE_PROBE_DIR"):
        raise RuntimeError("Run from isolated overlay with FORCE_PROBE_DIR set")
    if (root / "envs/_base_task.py").is_symlink():
        raise RuntimeError("Refusing original-source-backed _base_task.py")
    with (root / "task_config/demo_clean.yml").open() as f:
        args = yaml.safe_load(f)
    with (root / "task_config/_camera_config.yml").open() as f:
        cams = yaml.safe_load(f)
    with (root / "task_config/_embodiment_config.yml").open() as f:
        embodiments = yaml.safe_load(f)
    embodiment = embodiments[args["embodiment"][0]]["file_path"]
    with (root / embodiment / "config.yml").open() as f:
        robot_cfg = yaml.safe_load(f)
    args.update(task_name="lift_pot", task_config="demo_clean", policy_name="probe_only",
                ckpt_setting="none", left_robot_file=embodiment, right_robot_file=embodiment,
                left_embodiment_config=robot_cfg, right_embodiment_config=robot_cfg,
                dual_arm_embodied=True, eval_mode=True, eval_video_log=False,
                render_freq=0, save_data=False, save_path="/tmp/phizju-tactile-smoke-frames",
                head_camera_h=cams[args["camera"]["head_camera_type"]]["h"],
                head_camera_w=cams[args["camera"]["head_camera_type"]]["w"])
    task = lift_pot()
    try:
        task.setup_demo(now_ep_num=0, seed=100001, is_test=True, **args)
        before = task.get_obs()
        finger = task.robot.left_gripper[0][0].child_link
        builder = task.scene.create_actor_builder()
        builder.add_box_collision(half_size=[.035, .035, .035])
        cube = builder.build_kinematic(name="probe_cube")
        cube.set_pose(sapien.Pose(p=finger.get_entity_pose().p))
        action = np.asarray(before["joint_action"]["vector"], dtype=float)
        task.take_action(action)
        after = task.get_obs()
        result = {"task":"lift_pot", "before": before.get("tactile_force"),
                  "after": after.get("tactile_force"),
                  "action_shape":list(action.shape), "eval_success": bool(task.eval_success),
                  "note":"forced overlap cube to test physics-step channel; not success benchmark"}
        out = Path(os.environ["FORCE_PROBE_DIR"])/"task_smoke_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not result["after"] or not result["after"]["valid"]:
            raise RuntimeError("No valid force observation after task.take_action")
    finally:
        task.close_env()


if __name__ == "__main__": main()

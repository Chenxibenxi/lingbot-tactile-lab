"""Collect ONE fresh paired RoboTwin expert demonstration in an isolated overlay.

No existing HDF5/seed cache is reused. Images follow the original RoboTwin
_save-picture→HDF5 path; tactile/clock/joint measurements are independent sidecars.
Requires FORCE_PROBE_EXPERT_DIR and --out both inside a new private output root.
"""
import argparse
import json
import os
import socket
from pathlib import Path

from validate_expert_data import validate


def evaluate_outcome(robot):
    """Record planner and goal separately; a failed planner must not skip goal check."""
    planner_ok=bool(robot.plan_success)
    try:
        return planner_ok,bool(robot.check_success()),None
    except Exception as exc:
        return planner_ok,False,type(exc).__name__


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--episode",type=int,default=0)
    parser.add_argument("--task",default="lift_pot",choices=["lift_pot", "click_bell"])
    args=parser.parse_args()
    root=Path.cwd()
    if not (root/"envs/_base_task.py").is_file() or (root/"envs/_base_task.py").is_symlink():
        raise RuntimeError("run ONLY from a created expert overlay")
    if "record_saved_frame" not in (root/"envs/_base_task.py").read_text():
        raise RuntimeError("this overlay is not instrumented for expert images")
    output=args.out.resolve()
    source=Path(os.environ.get("ROBOTWIN_SOURCE", "/mnt/relaydata/lingbot/RoboTwin")).resolve()
    if root.resolve() in output.parents or source in output.parents:
        raise ValueError("output must not be inside baseline or overlay")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to mix with existing output or retry")
    force_dir=output/"force"
    if os.environ.get("FORCE_PROBE_EXPERT_DIR") != str(force_dir):
        raise RuntimeError("FORCE_PROBE_EXPERT_DIR must equal <out>/force exactly")
    import importlib
    import yaml
    with (root/"task_config/demo_clean.yml").open() as f:
        config=yaml.safe_load(f)
    with (root/"task_config/_camera_config.yml").open() as f:
        cams=yaml.safe_load(f)
    with (root/"task_config/_embodiment_config.yml").open() as f:
        embodiments=yaml.safe_load(f)
    embodiment=embodiments[config["embodiment"][0]]["file_path"]
    with (root/embodiment/"config.yml").open() as f:
        robot_config=yaml.safe_load(f)
    config.update(task_name=args.task,task_config="demo_clean",policy_name="expert",
        ckpt_setting="none",left_robot_file=embodiment,right_robot_file=embodiment,
        left_embodiment_config=robot_config,right_embodiment_config=robot_config,
        dual_arm_embodied=True,eval_mode=False,eval_video_log=False,render_freq=0,
        save_data=True,need_plan=True,save_path=str(output/"demonstration"),
        head_camera_h=cams[config["camera"]["head_camera_type"]]["h"],
        head_camera_w=cams[config["camera"]["head_camera_type"]]["w"])
    task=importlib.import_module("envs."+args.task)
    robot= getattr(task,args.task)()
    closed=False; probe=None; raw=None
    try:
        robot.setup_demo(now_ep_num=args.episode,seed=args.seed,**config)
        probe=robot._force_probe
        if probe is None:
            raise RuntimeError("expert probe did not initialize; refusing empty demonstration")
        robot.play_once()
        planner_ok,goal_ok,goal_check_error=evaluate_outcome(robot)
        success=planner_ok and goal_ok
        raw=probe.raw_path
        frames=probe.saved_frames
        if probe.failed or raw is None or frames<2 or not success:
            failure={"schema":"phizju_expert_trial_rejected_v1", "task":args.task,
                     "seed":args.seed,"episode":args.episode,
                     "source_machine_role":os.environ.get("TACTILE_SOURCE_MACHINE_ROLE","unverified"),
                     "source_hostname":socket.gethostname(),"planner_ok":planner_ok,
                     "goal_ok":goal_ok,"goal_check_error_type":goal_check_error,
                     "probe_failed":probe.failed,"saved_frame_markers":frames,
                     "raw_basename":raw.name if raw else None,
                     "accepted_for_training":False}
            with (output/"failure.json").open("x") as file:
                json.dump(failure,file,indent=2)
            raise RuntimeError(f"expert trial rejected: planner_ok={planner_ok}, goal_ok={goal_ok}, probe_failed={probe.failed}, frames={frames}")
        robot.close_env(); closed=True
        # Original RoboTwin conversion creates the actual HDF5 and preview video.
        robot.merge_pkl_to_hdf5_video()
        hdf5=output/"demonstration/data"/f"episode{args.episode}.hdf5"
        result=validate(raw,raw.with_suffix(".obs.jsonl"),raw.with_suffix(".frames.jsonl"),hdf5)
        completion={"schema":"phizju_expert_episode_complete_v1","success":True,
            "task":args.task,"seed":args.seed,"episode":args.episode,
            "source_machine_role":os.environ.get("TACTILE_SOURCE_MACHINE_ROLE", "unverified"),
            "source_hostname":socket.gethostname(),
            "source":"same-run RoboTwin expert; original RGB/state HDF5 and separate tactile markers",
            **result}
        dest=output/"completion.json"
        with dest.with_suffix(".tmp").open("x") as file:
            json.dump(completion,file,indent=2)
        dest.with_suffix(".tmp").replace(dest)
        validate(raw,raw.with_suffix(".obs.jsonl"),raw.with_suffix(".frames.jsonl"),hdf5,dest)
        print("EXPERT_PAIRED_VALIDATION_OK",json.dumps({k:result[k] for k in
            ("task","seed","episode","physics_steps","camera_observations","saved_frames","saved_last_step")}))
    finally:
        if not closed:
            robot.close_env()


if __name__=="__main__":main()

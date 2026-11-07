"""Collect ONE open-grasp negative in an isolated relay simulation overlay.

Never calls an expert-success gate, never emits completion.json, and refuses
to claim a negative unless the planner works and the task goal fails.
"""
import argparse
import importlib
import json
import os
import socket
from pathlib import Path

from controlled_negative import install_open_grasp_control,check_intervention
from record_expert_demo import evaluate_outcome
from validate_expert_data import validate,sha256,read_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--task',choices=['pick_diverse_bottles'],required=True)
    args=p.parse_args()
    if args.seed not in (100000,100007) or args.episode!=0:
        raise ValueError('bounded intervention only supports preregistered relay seeds episode0')
    base=Path('/mnt/relaydata/lingbot-tactile-lab').resolve(strict=True)
    output=args.out.resolve()
    allowed={100000:'relay_pickdiverse_s100000_open_grasp1',
             100007:'relay_pickdiverse_s100007_open_grasp_spatial1'}
    if output.parent!=base/'data/negative' or output.name!=allowed[args.seed] or output.exists():
        raise ValueError('fresh preregistered isolated data/negative trial required')
    spatial=args.seed==100007
    expected_geometry=str(output/'geometry') if spatial else None
    if os.environ.get('TACTILE_CONTACT_GEOMETRY_DIR')!=expected_geometry:
        raise ValueError('spatial world contact opt-in must match only the preregistered new trial')
    overlay=Path.cwd().resolve(strict=True)
    if (overlay/'envs/_base_task.py').is_symlink() or 'record_saved_frame' not in (overlay/'envs/_base_task.py').read_text():
        raise ValueError('private instrumented expert overlay required')
    force_dir=output/'force'
    if os.environ.get('FORCE_PROBE_EXPERT_DIR')!=str(force_dir) or os.environ.get('TACTILE_SOURCE_MACHINE_ROLE')!='relay_single_rtx3090':
        raise ValueError('negative ForceProbe/source role not isolated')
    import yaml
    with (overlay/'task_config/demo_clean.yml').open() as f:config=yaml.safe_load(f)
    with (overlay/'task_config/_camera_config.yml').open() as f:cameras=yaml.safe_load(f)
    with (overlay/'task_config/_embodiment_config.yml').open() as f:embodiments=yaml.safe_load(f)
    embodiment=embodiments[config['embodiment'][0]]['file_path']
    with (overlay/embodiment/'config.yml').open() as f:robot_config=yaml.safe_load(f)
    config.update(task_name=args.task,task_config='demo_clean',policy_name='expert',
        ckpt_setting='none',left_robot_file=embodiment,right_robot_file=embodiment,
        left_embodiment_config=robot_config,right_embodiment_config=robot_config,
        dual_arm_embodied=True,eval_mode=False,eval_video_log=False,render_freq=0,
        save_data=True,need_plan=True,save_path=str(output/'demonstration'),
        head_camera_h=cameras[config['camera']['head_camera_type']]['h'],
        head_camera_w=cameras[config['camera']['head_camera_type']]['w'])
    task=importlib.import_module('envs.'+args.task)
    robot=getattr(task,args.task)()
    closed=False;raw=None
    try:
        robot.setup_demo(now_ep_num=args.episode,seed=args.seed,**config)
        probe=robot._force_probe
        if probe is None:raise RuntimeError('no expert ForceProbe attached')
        calls=install_open_grasp_control(robot)
        robot.play_once()
        planner_ok,goal_ok,goal_error=evaluate_outcome(robot)
        raw=probe.raw_path;frames=probe.saved_frames
        if probe.failed or raw is None or frames<2 or goal_error is not None:
            raise RuntimeError('negative probe/frames/goal check invalid')
        check_intervention(calls,planner_ok,goal_ok)
        robot.close_env();closed=True
        source={key:sha256(path) for key,path in
                (('raw',raw),('obs',raw.with_suffix('.obs.jsonl')),
                 ('frames',raw.with_suffix('.frames.jsonl')))}
        geometry=output/'geometry'/f'{raw.stem}.geometry.jsonl'
        if spatial:
            from validate_contact_geometry import validate_pair
            pair=validate_pair(raw,geometry)
            if pair['steps']<=0:
                raise ValueError('spatial geometry did not cover this physics run')
            source['geometry']=sha256(geometry)
        gate={'schema':'lingbot_controlled_open_grasp_gate_v1',
              'status':'controlled_negative_not_training_approved',
              'task':args.task,'seed':args.seed,'episode':args.episode,
              'source_machine_role':'relay_single_rtx3090','source_hostname':socket.gethostname(),
              'perturbation':'grasp_actor gripper_pos=1.0 on two arms; no baseline RoboTwin edit',
              'intervened_grasps':len(calls),'planner_ok':True,'goal_ok':False,
              'saved_frame_markers':frames,'source_sha256':source,
              'accepted_for_expert_training':False}
        with (output/'negative_gate.json').open('x') as file:json.dump(gate,file,indent=2)
        robot.merge_pkl_to_hdf5_video()  # same-run private cache, but *no* success completion
        hdf5=output/'demonstration/data'/f'episode{args.episode}.hdf5'
        checked=validate(raw,raw.with_suffix('.obs.jsonl'),raw.with_suffix('.frames.jsonl'),hdf5)
        if spatial:
            from validate_contact_geometry import validate_pair
            validate_pair(raw,geometry)
            checked['sha256']['geometry']=sha256(geometry)
        _,*physics=read_jsonl(raw)
        contacts={side:sum(int(r[side]['points']>0) for r in physics) for side in ('left','right')}
        manifest={'schema':'lingbot_controlled_negative_complete_v1',
                  'status':'quarantined_negative_not_training_or_eval',
                  'task':args.task,'seed':args.seed,'episode':args.episode,
                  'source_machine_role':'relay_single_rtx3090','source_hostname':socket.gethostname(),
                  'planner_ok':True,'goal_ok':False,'intervened_grasps':len(calls),
                  'negative_gate_sha256':sha256(output/'negative_gate.json'),
                  'contact_physics_steps_by_side':contacts,
                  'accepted_for_expert_training':False,**checked}
        dest=output/'negative_completion.json'
        with dest.with_suffix('.tmp').open('x') as file:json.dump(manifest,file,indent=2)
        dest.with_suffix('.tmp').replace(dest)
        print('CONTROLLED_NEGATIVE_DATA_VALIDATED',json.dumps({k:manifest[k] for k in
              ('physics_steps','saved_frames','planner_ok','goal_ok','contact_physics_steps_by_side')}))
    finally:
        if not closed:robot.close_env()


if __name__=='__main__':main()

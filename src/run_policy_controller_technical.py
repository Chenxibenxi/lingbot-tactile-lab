"""One fresh known-seed TECHNICAL qpos replay through original take_action.

Old same-seed expert drive targets are only control waypoints; new run's images,
contacts and actual controller actions never become approved training examples.
Requires a private isolated policy-equivalent expert overlay and external GPU
lease/idle-Strata stop+restore wrapper. No HDF5 conversion or RPC/model train.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
import socket
from pathlib import Path
from expert_command_timeline import validate_commands
from patch_policy_equivalent_expert_overlay import SOURCE,ORIGINAL_SHA,generate
from technical_take_action_waypoints import propose_waypoints,sanitized_card
from technical_take_action_replay import CAP_8000_TRIAL,run_controller
from validate_expert_data import read_jsonl,sha256,validate

REPO=Path(__file__).resolve().parents[1]
TRIAL='relay_knownseed100013_h50_controller_technical_v1'
TRIAL_DIAG='relay_knownseed100013_h50_controller_xyz_diagnostic_v2'
TRIAL_DIAG_V3='relay_knownseed100013_h50_controller_xyz_repair_v3'
TRIAL_CAP_V4=CAP_8000_TRIAL
SOURCE_TRIAL=REPO/'data/paired/relay_pickdiverse_s100013_command_smoke_expert0'


def validate_source_proposal():
    """Recheck pinned old technical sources without authorizing another trial."""
    if sha256(SOURCE)!=ORIGINAL_SHA:
        raise ValueError('original RoboTwin source changed')
    files=list((SOURCE_TRIAL/'commands').glob('*.commands.jsonl'))
    if len(files)!=1 or files[0].is_symlink():raise ValueError('missing unique source command sidecar')
    path=files[0]
    with path.open(encoding='utf-8') as file:
        header=json.loads(file.readline())
    raw=SOURCE_TRIAL/'force'/header['probe_file']
    frames=raw.with_suffix('.frames.jsonl')
    source_audit=validate_commands(raw,frames,path)
    completion=SOURCE_TRIAL/'completion.json'
    hdf5=SOURCE_TRIAL/'demonstration/data/episode0.hdf5'
    validate(raw,raw.with_suffix('.obs.jsonl'),frames,hdf5,completion)
    fixed=json.loads((REPO/'datasets/relay_known_seed_h50_waypoint_feasibility_v1.json').read_text())
    if (fixed.get('source_raw_sha256')!=source_audit['source_sha256']['raw'] or
            fixed.get('source_command_sha256')!=source_audit['source_sha256']['commands'] or
            fixed.get('training_approval') is not False):
        raise ValueError('prior successful technical source no longer matches frozen SHA card')
    proposal=propose_waypoints(read_jsonl(path)[1:],
        raw_sha256=source_audit['source_sha256']['raw'],
        command_sha256=source_audit['source_sha256']['commands'])
    if proposal['proposed_waypoint_count']!=64 or source_audit['physics_steps']!=1582:
        raise ValueError('known-seed technical waypoints/source clock changed')
    return proposal,source_audit


def validate_failed_xyz_v2():
    """Require that the immutable first xyz attempt really failed at 37 ticks."""
    trial=REPO/'data/technical_replay'/TRIAL_DIAG
    card=json.loads((REPO/'datasets/relay_known_seed_bottle_xyz_instrumentation_failure_v1.json').read_text())
    if (card.get('status')!='instrumentation_failed_after_first_real_action_before_xyz_record' or
            card.get('physics_steps')!=37 or card.get('training_approval') is not False):
        raise ValueError('prior xyz instrumentation failure card changed')
    matches=list((trial/'force').glob('*.jsonl'))
    raw=[p for p in matches if not p.name.endswith(('.obs.jsonl','.frames.jsonl'))]
    if len(raw)!=1 or trial.is_symlink():
        raise ValueError('prior xyz trial not uniquely preserved')
    paths={'raw':raw[0],'frames':raw[0].with_suffix('.frames.jsonl'),
           'calls':trial/'calls.jsonl','geometry':trial/'bottle_xyz_after_action.jsonl',
           'failure':trial/'technical_failure.json'}
    if any(sha256(path)!=card['source_file_sha256'][name] for name,path in paths.items()):
        raise ValueError('prior partial xyz trial SHA changed; refuse repair retry')
    return True


def validate_fixed_cap_xyz_v3():
    """Only extend the exact observed failure, never replace or reinterpret it."""
    trial=REPO/'data/technical_replay'/TRIAL_DIAG_V3
    card=json.loads((REPO/'datasets/relay_known_seed_h50_bottle_xyz_fixed_cap_diagnosis_v1.json').read_text())
    if (card.get('status')!='goal_failed_both_y_distance_outside_0_1_m_at_cap' or
            card.get('actual_take_action_calls')!=54 or card.get('physics_steps')!=6018 or
            card.get('training_approval') is not False or trial.is_symlink()):
        raise ValueError('prior fixed-cap xyz failure card changed')
    matches=list((trial/'force').glob('*.jsonl'))
    raw=[p for p in matches if not p.name.endswith(('.obs.jsonl','.frames.jsonl'))]
    if len(raw)!=1:raise ValueError('prior fixed-cap xyz trial not uniquely preserved')
    paths={'raw':raw[0],'frames':raw[0].with_suffix('.frames.jsonl'),
           'obs':raw[0].with_suffix('.obs.jsonl'),'calls':trial/'calls.jsonl',
           'geometry':trial/'bottle_xyz_after_action.jsonl',
           'result':trial/'technical_result.json'}
    if any(sha256(path)!=card['source_file_sha256'][name] for name,path in paths.items()):
        raise ValueError('prior fixed-cap xyz trial SHA changed; refuse longer technical pilot')
    return True


def validate_source_and_paths(out,*,overlay=None,diagnostic=False,repair_v3=False,cap_v4=False):
    out=Path(out).resolve(strict=False)
    if repair_v3 and not diagnostic:
        raise ValueError('v3 requires user-approved xyz diagnostic mode')
    if cap_v4 and (not diagnostic or repair_v3):
        raise ValueError('v4 requires unique user-approved xyz cap8000 mode')
    expected=TRIAL_CAP_V4 if cap_v4 else (TRIAL_DIAG_V3 if repair_v3 else (TRIAL_DIAG if diagnostic else TRIAL))
    if (socket.gethostname()!='admin001-MS-7D90' or out.name!=expected or
            out.parent!=REPO/'data/technical_replay' or out.exists()):
        raise ValueError('only one fresh relay-known-seed technical replay output')
    if repair_v3:validate_failed_xyz_v2()
    if cap_v4:validate_fixed_cap_xyz_v3()
    proposal,source_audit=validate_source_proposal()
    if overlay is not None:
        overlay=Path(overlay).resolve(strict=True)
        patched=overlay/'envs/_base_task.py'
        expected=hashlib.sha256(generate(SOURCE.read_text()).encode()).hexdigest()
        if (overlay!=REPO/'work/overlay_policy_equivalent_pilot_v1' or
                not patched.is_file() or patched.is_symlink() or sha256(patched)!=expected):
            raise ValueError('not the SHA-locked private controller overlay')
    return out,proposal,source_audit


def bottle_goal_geometry(robot):
    """Read task functional points AFTER a real action; never model input."""
    result={}
    for tag,actor,target in (('left',robot.bottle1,robot.left_target_pose),
                             ('right',robot.bottle2,robot.right_target_pose)):
        point=actor.get_functional_point(0)  # RoboTwin list is xyz + quaternion (7D)
        if point is None or len(point)<3:
            raise ValueError('missing bottle functional point in private diagnostic')
        xyz=[float(v) for v in point[:3]]
        if any(not math.isfinite(v) for v in xyz):
            raise ValueError('nonfinite bottle xyz in private diagnostic')
        dx=xyz[0]-float(target[0]);dy=xyz[1]-float(target[1])
        result[tag]={'functional_point_xyz':xyz,'target_delta_xy':[dx,dy],
                     'xy_within_task_gate':abs(dx)<0.1 and abs(dy)<0.1,
                     'z_above_task_gate':xyz[2]>0.89}
    return result


def run(out,overlay,*,diagnostic=False,repair_v3=False,cap_v4=False):
    out,proposal,audit=validate_source_and_paths(out,overlay=overlay,
                               diagnostic=diagnostic,repair_v3=repair_v3,cap_v4=cap_v4)
    if (os.environ.get('TACTILE_POLICY_EQUIVALENT_EXPERT')!='1' or
            os.environ.get('FORCE_PROBE_EXPERT_DIR')!=str(out/'force')):
        raise RuntimeError('fresh technical probe flags/output not set')
    if Path.cwd().resolve()!=Path(overlay).resolve():
        raise RuntimeError('must import RoboTwin from private overlay working directory')
    import importlib
    import yaml
    with (Path.cwd()/'task_config/demo_clean.yml').open() as f:config=yaml.safe_load(f)
    with (Path.cwd()/'task_config/_camera_config.yml').open() as f:cams=yaml.safe_load(f)
    with (Path.cwd()/'task_config/_embodiment_config.yml').open() as f:embodiments=yaml.safe_load(f)
    embodiment=embodiments[config['embodiment'][0]]['file_path']
    with (Path.cwd()/embodiment/'config.yml').open() as f:robot_config=yaml.safe_load(f)
    config.update(task_name='pick_diverse_bottles',task_config='demo_clean',policy_name='expert',
        ckpt_setting='none',left_robot_file=embodiment,right_robot_file=embodiment,
        left_embodiment_config=robot_config,right_embodiment_config=robot_config,
        dual_arm_embodied=True,eval_mode=False,eval_video_log=False,render_freq=0,
        save_data=True,need_plan=True,save_path=str(out/'demonstration'),
        head_camera_h=cams[config['camera']['head_camera_type']]['h'],
        head_camera_w=cams[config['camera']['head_camera_type']]['w'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=False,exist_ok=False)
    source_card=sanitized_card(proposal)
    with (out/'proposal.json').open('x') as file:json.dump(source_card,file,indent=2)
    task=importlib.import_module('envs.pick_diverse_bottles')
    robot=getattr(task,'pick_diverse_bottles')();initialized=False
    try:
        robot.setup_demo(now_ep_num=0,seed=100013,**config);initialized=True
        probe=robot._force_probe
        if probe is None or probe.step!=0:
            raise RuntimeError('fresh expert probe not initialized at step zero')
        def capture(current):
            before=current._force_probe.step
            last=current.FRAME_IDX
            current._take_picture()  # NEW-run RGB pickle + saved frame marker only
            if current._force_probe.failed or current.FRAME_IDX!=last+1:
                raise RuntimeError('same-run current camera capture failed')
            return {'capture_physics_step':before,'frame_index':last}
        with ExitStack() as stack:
            if diagnostic:
                geometry=stack.enter_context((out/'bottle_xyz_after_action.jsonl').open('x',encoding='utf-8'))
                original_take_action=robot.take_action
                def measure_real_action(command,action_type='qpos'):
                    response=original_take_action(command,action_type=action_type)
                    geometry.write(json.dumps({'call_index':robot.take_action_cnt-1,
                        'after_physics_step':robot._force_probe.step,
                        'bottles':bottle_goal_geometry(robot),
                        'role':'after_action_diagnostic_only_not_model_input'},allow_nan=False)+'\n')
                    geometry.flush()
                    return response
                robot.take_action=measure_real_action
            result=run_controller(robot,proposal,output_path=out/'calls.jsonl',capture=capture,
                                  physics_cap=8000 if cap_v4 else 6000)
        result['task']='pick_diverse_bottles';result['seed']=100013
        result['source_machine_role']='relay_single_rtx3090'
        result['source_command_sha256']=audit['source_sha256']['commands']
        result['new_trial_id']=TRIAL_CAP_V4 if cap_v4 else (TRIAL_DIAG_V3 if repair_v3 else (TRIAL_DIAG if diagnostic else TRIAL))
        result['physics_cap_authorized']=8000 if cap_v4 else 6000
        result['diagnostic_geometry_private']=bool(diagnostic)
        result['goal_check_after_run']=bool(robot.check_success())
        result['training_approval']=False
        with (out/'technical_result.json').open('x') as file:json.dump(result,file,indent=2)
        print('H50_TECHNICAL_REPLAY_SUMMARY',json.dumps({key:result[key] for key in
              ('actual_take_action_calls','physics_steps','stopped_reason','goal_check_after_run','training_approval')}),flush=True)
        return 0 if result['goal_success_reported'] and result['goal_check_after_run'] and result['actual_take_action_calls']>=50 else 7
    except Exception as exc:
        with (out/'technical_failure.json').open('x') as file:
            json.dump({'status':'technical_trial_failed_not_training_approved',
                       'exception_type':type(exc).__name__,'physics_steps':getattr(getattr(robot,'_force_probe',None),'step',None),
                       'training_approval':False},file)
        raise
    finally:
        if initialized:robot.close_env()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--overlay',type=Path,default=None)
    p.add_argument('--run',action='store_true')
    p.add_argument('--diagnostic-xyz-v2',action='store_true',
                   help='one used/immutable first diagnostic id; refuse reuse')
    p.add_argument('--diagnostic-xyz-repair-v3',action='store_true',
                   help='one used/immutable fixed-cap 7D bottle diagnostic id')
    p.add_argument('--technical-cap8000-v4',action='store_true',
                   help='one user-approved same-seed 64-waypoint technical cap extension, never training')
    args=p.parse_args()
    if sum((args.diagnostic_xyz_v2,args.diagnostic_xyz_repair_v3,args.technical_cap8000_v4))>1:
        p.error('technical trial versions cannot be combined')
    diagnostic=args.diagnostic_xyz_v2 or args.diagnostic_xyz_repair_v3 or args.technical_cap8000_v4
    validate_source_and_paths(args.out,overlay=args.overlay if args.run else None,
                              diagnostic=diagnostic,repair_v3=args.diagnostic_xyz_repair_v3,
                              cap_v4=args.technical_cap8000_v4)
    if not args.run:
        print('TECHNICAL_SOURCE_64_WAYPOINTS_VERIFIED_NO_GPU_OR_OUTPUT')
    else:
        if args.overlay is None:p.error('--run requires isolated overlay')
        raise SystemExit(run(args.out,args.overlay,diagnostic=diagnostic,
                             repair_v3=args.diagnostic_xyz_repair_v3,
                             cap_v4=args.technical_cap8000_v4))


if __name__=='__main__':main()

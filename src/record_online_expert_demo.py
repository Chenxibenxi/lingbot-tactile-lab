"""Isolated one-seed online-rule HIGH-LEVEL expert collector; dry-run by default.

Never use old drive-target waypoints or policy self-rollouts. The only --run
entry requires an ancestor-shell GPU lease, the SHA-locked overlay, explicit
bounded user approval and no competing compute process. A success completion
is only a candidate; H50/source/norm approval is a separate later review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib import error,request

from online_expert_pilot_executor import execute_online_expert,MAX_PHYSICS_STEPS
from patch_policy_equivalent_expert_overlay import generate
from preregister_online_expert_trial import REPO,SEED,TASK,TRIAL,SOURCE,preregister
from validate_expert_data import read_jsonl,sha256,validate

OVERLAY=REPO/'work/overlay_policy_equivalent_pilot_v1'
LEASE_TOOL=Path('/home/admin001/ai-video-studio/tools/gpu_lease.py')
LEASE_OWNER='lingbot-tactile-lab/online-expert-s100022-v1'


def verify_fresh_source(out, overlay=OVERLAY):
    """No directory is created and no GPU is touched."""
    out=Path(out).resolve(strict=False)
    if (socket.gethostname()!='admin001-MS-7D90' or
            out!=REPO/'data/paired'/TRIAL or out.exists() or
            (out.parent/(TRIAL+'.lock')).exists() or (out.parent/(TRIAL+'.status')).exists()):
        raise ValueError('only one fresh relay s100022 online-rule expert trial')
    manifest=preregister()
    overlay=Path(overlay).resolve(strict=True)
    patched=overlay/'envs/_base_task.py'
    expected=hashlib.sha256(generate(SOURCE.read_text()).encode()).hexdigest()
    if (overlay!=OVERLAY or not patched.is_file() or patched.is_symlink() or
            sha256(patched)!=expected or 'record_saved_frame' not in patched.read_text()):
        raise ValueError('online pilot requires the exact isolated original-controller force overlay')
    return manifest


def _ancestors():
    pid=os.getpid();parents=set()
    for _ in range(3):
        lines=(Path('/proc')/str(pid)/'status').read_text().splitlines()
        parent=next((int(row.split(':',1)[1].strip()) for row in lines if row.startswith('PPid:')),0)
        if parent<=1:break
        parents.add(parent);pid=parent
    return parents


def verify_gpu_lease():
    """Reject an unleased direct --run even if a GPU appears empty."""
    info=json.loads(subprocess.run([sys.executable,str(LEASE_TOOL),'status','--json'],
                        capture_output=True,text=True,check=True,timeout=10).stdout)
    lease=info.get('lease') or {}
    if (info.get('state')!='held' or info.get('holder')!=LEASE_OWNER or
            info.get('pid_alive') is not True or lease.get('pid') not in _ancestors() or
            info.get('remaining_s',0)<600 or
            any(w.get('owner')!=LEASE_OWNER for w in info.get('waiting',[]))):
        raise RuntimeError('online expert requires live own ancestor lease TTL>=600/no waiter')
    compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                       capture_output=True,text=True,check=True,timeout=10).stdout
    free=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if compute.strip() or not free.isdecimal() or int(free)<16000:
        raise RuntimeError('online expert requires 16GiB free and no GPU compute workload')
    try:
        with request.urlopen('http://127.0.0.1:8080/health',timeout=3) as response:health=json.load(response)
        if health.get('service')!='strata' or health.get('loaded') is not False:
            raise RuntimeError('Strata must be unloaded before allocating the one GPU')
    except error.URLError as exc:
        if not isinstance(exc.reason,ConnectionRefusedError):
            raise RuntimeError('ambiguous Strata health, refuse GPU trial') from exc
    return True


def audit_same_run_online_clock(raw,actions,physics_steps):
    """Reject missing camera anchors/overlapping high-level actions pre-completion."""
    raw=Path(raw)
    physics=read_jsonl(raw)
    cameras=read_jsonl(raw.with_suffix('.frames.jsonl'))
    observations=read_jsonl(raw.with_suffix('.obs.jsonl'))
    if (len(physics)-1!=physics_steps or len(cameras)-1!=len(actions) or
            len(actions)<50 or len(observations)<len(actions)+1):
        raise ValueError('online action count and same-run camera/physics counts disagree')
    prior=None
    for index,(row,frame) in enumerate(zip(actions,cameras[1:])):
        before=row['before_physics_step'];after=row['after_physics_step']
        obs_index=frame.get('obs_index')
        if (row['action_index']!=index or row['camera_frame_index']!=index or
                frame.get('frame_index')!=index or frame.get('physics_step')!=before or
                row.get('capture_physics_step')!=before or
                type(before) is not int or type(after) is not int or
                before<0 or after<=before or after>physics_steps or
                (prior is not None and prior!=before) or
                physics[after].get('physics_step')!=after or
                type(obs_index) is not int or obs_index<0 or obs_index+1>=len(observations) or
                observations[obs_index+1].get('physics_step')!=before or
                row.get('controller_method')!='Base_Task.take_action' or
                row.get('training_approval') is not False):
            raise ValueError('new-run online action/RGB/observation/physics interval mismatch')
        prior=after
    if prior!=physics_steps or actions[-1].get('observed_success') is not True:
        raise ValueError('new-run last online action did not end at verified goal step')
    return {'real_highlevel_call_count':len(actions),
            'source_checked_same_run_clock_only':True,'training_approval':False}


def _config(root,out):
    import yaml
    with (root/'task_config/demo_clean.yml').open() as f:cfg=yaml.safe_load(f)
    with (root/'task_config/_camera_config.yml').open() as f:cams=yaml.safe_load(f)
    with (root/'task_config/_embodiment_config.yml').open() as f:emb=yaml.safe_load(f)
    robot_path=emb[cfg['embodiment'][0]]['file_path']
    with (root/robot_path/'config.yml').open() as f:robot_cfg=yaml.safe_load(f)
    cfg.update(task_name=TASK,task_config='demo_clean',policy_name='expert',ckpt_setting='none',
               left_robot_file=robot_path,right_robot_file=robot_path,
               left_embodiment_config=robot_cfg,right_embodiment_config=robot_cfg,
               dual_arm_embodied=True,eval_mode=False,eval_video_log=False,render_freq=0,
               save_data=True,need_plan=True,save_path=str(out/'demonstration'),
               head_camera_h=cams[cfg['camera']['head_camera_type']]['h'],
               head_camera_w=cams[cfg['camera']['head_camera_type']]['w'])
    return cfg


def run_once(out,overlay):
    manifest=verify_fresh_source(out,overlay)
    if (os.environ.get('TACTILE_ONLINE_EXPERT_PILOT_APPROVED')!='1' or
            os.environ.get('TACTILE_POLICY_EQUIVALENT_EXPERT')!='1' or
            os.environ.get('TACTILE_SOURCE_MACHINE_ROLE')!='relay_single_rtx3090' or
            os.environ.get('FORCE_PROBE_EXPERT_DIR')!=str(Path(out)/'force') or
            Path.cwd().resolve()!=Path(overlay).resolve()):
        raise RuntimeError('missing explicit one-seed expert pilot flags/isolated overlay')
    verify_gpu_lease()  # no imports of sapien/CUDA before this point
    import importlib
    out=Path(out);out.mkdir(parents=False,exist_ok=False)
    with (out/'preregistration.json').open('x') as f:json.dump(manifest,f,indent=2)
    robot=getattr(importlib.import_module('envs.'+TASK),TASK)();closed=False;probe=None
    try:
        robot.setup_demo(now_ep_num=0,seed=SEED,**_config(Path.cwd(),out))
        probe=robot._force_probe
        if probe is None or probe.step!=0:raise RuntimeError('fresh expert force probe missing')
        def capture(task):
            before=task._force_probe.step;index=task.FRAME_IDX
            task._take_picture()
            if task._force_probe.failed or task.FRAME_IDX!=index+1:
                raise RuntimeError('same-run current RGB capture failed')
            return {'capture_physics_step':before,'frame_index':index}
        result=execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS)
        actions=result.pop('private_action_rows_in_memory_only')
        with (out/'online_actions.jsonl').open('x',encoding='utf-8') as f:
            f.write(json.dumps({'schema':'lingbot_online_highlevel_expert_trial_actions_v1',
                'seed':SEED,'task':TASK,'trial_id':TRIAL,'source_sha256':{
                'original_base_task':manifest['original_base_task_sha256'],
                'original_bottle_task':manifest['original_bottle_task_sha256'],
                'original_robot_planner':manifest['original_robot_planner_sha256']},
                'status':'private_online_candidate_not_training_approved'},allow_nan=False)+'\n')
            for row in actions:f.write(json.dumps(row,allow_nan=False)+'\n')
        raw=probe.raw_path;frames=probe.saved_frames
        planner_ok=bool(robot.plan_success)
        goal_ok=bool(robot.check_success())
        accepted=(planner_ok and goal_ok and result['goal_success_reported'] and
                  result['actual_take_action_calls']>=50 and frames>=50 and not probe.failed and
                  result['stopped_reason']=='early_goal_success')
        with (out/'pilot_result.json').open('x') as f:
            json.dump({**result,'seed':SEED,'task':TASK,'planner_ok':planner_ok,
                       'goal_ok':goal_ok,'source_machine_role':'relay_single_rtx3090',
                       'online_action_sha256':sha256(out/'online_actions.jsonl'),
                       'success_candidate_not_training_approved':accepted},f,indent=2)
        if not accepted:
            with (out/'failure.json').open('x') as f:
                json.dump({'schema':'lingbot_online_expert_pilot_rejected_v1','seed':SEED,
                           'planner_ok':planner_ok,'goal_ok':goal_ok,
                           'actual_take_action_calls':result['actual_take_action_calls'],
                           'physics_steps':result['physics_steps'],
                           'reason':result['stopped_reason'],'training_approval':False},f)
            return 7
        robot.close_env();closed=True
        clock_audit=audit_same_run_online_clock(raw,actions,result['physics_steps'])
        gate=out/'expert_success_gate.json'
        gate_data={'schema':'phizju_expert_success_gate_v1','success':True,
                   'task':TASK,'seed':SEED,'episode':0,
                   'source_machine_role':'relay_single_rtx3090',
                   'source_hostname':socket.gethostname(),
                   'planner_ok':planner_ok,'goal_ok':goal_ok,
                   'saved_frame_markers':frames,'online_action_sha256':sha256(out/'online_actions.jsonl'),
                   'online_clock_audit':clock_audit,
                   'source_sha256':{k:sha256(p) for k,p in
                     (('raw',raw),('obs',raw.with_suffix('.obs.jsonl')),
                      ('frames',raw.with_suffix('.frames.jsonl')))}}
        with gate.open('x') as f:json.dump(gate_data,f,indent=2)
        robot.merge_pkl_to_hdf5_video()
        hdf5=out/'demonstration/data/episode0.hdf5'
        source=validate(raw,raw.with_suffix('.obs.jsonl'),raw.with_suffix('.frames.jsonl'),hdf5)
        completion={'schema':'phizju_expert_episode_complete_v1','success':True,
                    'task':TASK,'seed':SEED,'episode':0,
                    'source_machine_role':'relay_single_rtx3090',
                    'source_hostname':socket.gethostname(),
                    'source':'new same-run online rule expert; highlevel calls require later H50 audit',
                    'online_action_sha256':gate_data['online_action_sha256'],
                    'success_gate_sha256':sha256(gate),**source}
        # Do not leave a successful completion marker if the final provenance
        # check fails: the temporary manifest itself can be validated in place.
        pending=out/'completion.json.tmp'
        with pending.open('x') as f:json.dump(completion,f,indent=2)
        validate(raw,raw.with_suffix('.obs.jsonl'),raw.with_suffix('.frames.jsonl'),hdf5,
                 pending)
        pending.replace(out/'completion.json')
        print('ONLINE_EXPERT_FRESH_SUCCESS_CANDIDATE',json.dumps({
            'seed':SEED,'physics_steps':result['physics_steps'],
            'actual_take_action_calls':result['actual_take_action_calls'],
            'training_approval':False}),flush=True)
        return 0
    except Exception as exc:
        if not (out/'failure.json').exists() and not (out/'completion.json').exists():
            with (out/'failure.json').open('x') as f:
                json.dump({'schema':'lingbot_online_expert_pilot_rejected_v1','seed':SEED,
                           'exception_type':type(exc).__name__,
                           'physics_steps':getattr(probe,'step',None),
                           'training_approval':False},f)
        raise
    finally:
        if not closed:robot.close_env()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--overlay',type=Path,default=OVERLAY)
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    pre=verify_fresh_source(args.out,args.overlay)
    if not args.run:
        print(json.dumps({k:pre[k] for k in ('seed','new_trial_id','gpu_run_allowed_now','training_approval')},sort_keys=True))
    else:raise SystemExit(run_once(args.out,args.overlay))


if __name__=='__main__':main()

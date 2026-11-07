"""Opt-in same-run recovery collector; requires a separate frozen readiness manifest."""
import argparse
import importlib
import json
import os
from pathlib import Path
import socket
from recovery_action_journal import RecoveryJournal
from recovery_expert_executor import execute_recovery
from recovery_clock_audit import audit_clock
from validate_expert_data import validate,read_jsonl,sha256

R=Path(__file__).resolve().parents[1]
OVERLAY=R/'work/overlay_policy_equivalent_pilot_v1'
PROTOCOL=R/'docs/RECOVERY_DEVELOPMENT_PROTOCOL_V1_ZH.md'
CONDITIONS=('expert_control','left_open_once_recovery')


def require_attempt_identity(source,execution,seed):
    if source['task']!='pick_diverse_bottles' or source['seed']!=seed or source['episode']!=0:
        raise ValueError('recorded source does not belong to the registered attempt')
    if source['physics_steps']!=execution['physics_steps'] or source['saved_frames']!=execution['action_count']:
        raise ValueError('recorded source does not cover exactly the executed actions')


def preflight(seed,condition):
    if type(seed) is not int or seed not in range(100119,100123) or condition not in CONDITIONS:
        raise ValueError('fixed development matrix required')
    reservation=json.loads((R/'datasets/recovery_development_reservations_v1.json').read_text())
    if reservation['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('reserved protocol changed')
    manifest=json.loads((R/'data/reports/recovery_development_v1/manifest.json').read_text())
    if manifest.get('gpu_execution_ready') is not True or manifest.get('protocol_sha256')!=sha256(PROTOCOL):
        raise ValueError('recovery implementation not ready/frozen')
    required={'src/record_recovery_development.py','src/recovery_expert_executor.py',
              'src/recovery_action_journal.py','src/recovery_clock_audit.py','src/recovery_supervision_contract.py',
              'src/online_bottle_expert_actions.py','src/online_expert_pilot_executor.py',
              'src/record_online_expert_demo.py','src/validate_expert_data.py'}
    if not required.issubset(manifest['files']):raise ValueError('missing implementation source pins')
    for name,digest in manifest['files'].items():
        if sha256(R/name)!=digest:raise ValueError('recovery source changed: '+name)
    prerequisite=R/'data/reports/cwmj_m8/continuation_result.json'
    if not prerequisite.exists() or json.loads(prerequisite.read_text()).get('status')!='completed':
        raise ValueError('M8 training/evaluation must finish before recovery collection')
    if Path.cwd().resolve()!=OVERLAY:raise ValueError('isolated simulator cwd required')
    from patch_policy_equivalent_expert_overlay import generate
    from preregister_online_expert_trial import SOURCE,TASK_SOURCE,ROBOT_SOURCE,TASK_SOURCE_SHA,ROBOT_SOURCE_SHA
    if (OVERLAY/'envs/_base_task.py').read_text()!=generate(SOURCE.read_text()):raise ValueError('overlay drift')
    if sha256(TASK_SOURCE)!=TASK_SOURCE_SHA or sha256(ROBOT_SOURCE)!=ROBOT_SOURCE_SHA:raise ValueError('original task/controller drift')
    from probe_real_v2_bf16_stream import require_gpu_window
    require_gpu_window()
    return manifest


def run(seed,condition):
    manifest=preflight(seed,condition)
    out=R/f'data/paired/recovery_dev_{seed}_{condition}';out.mkdir(exist_ok=False)
    (out/'preregistration.json').write_text(json.dumps(manifest,indent=2))
    os.environ.update(FORCE_PROBE_EXPERT_DIR=str(out/'force'),TACTILE_POLICY_EQUIVALENT_EXPERT='1',
                      TACTILE_ONLINE_EXPERT_PILOT_APPROVED='1',TACTILE_SOURCE_MACHINE_ROLE='relay_single_rtx3090')
    from record_online_expert_demo import _config
    from cwmj_bottle_factory import install
    scenario=install();scenario['factory_sha256']=sha256(R/'src/cwmj_bottle_factory.py')
    (out/'scenario.json').write_text(json.dumps(scenario,indent=2))
    task=None;probe=None;execution=None;error=None;closed=False
    try:
        task=importlib.import_module('envs.pick_diverse_bottles').pick_diverse_bottles()
        task.setup_demo(now_ep_num=0,seed=seed,**_config(OVERLAY,out));probe=task._force_probe
        def capture(robot):
            before=probe.step;index=robot.FRAME_IDX;robot._take_picture()
            if probe.failed or robot.FRAME_IDX!=index+1:raise RuntimeError('same-run camera capture failed')
            return dict(capture_physics_step=before,frame_index=index)
        with RecoveryJournal(out/'action_journal.jsonl',seed=seed,condition=condition,protocol_sha256=sha256(PROTOCOL)) as journal:
            execution=execute_recovery(task,capture,perturb=condition=='left_open_once_recovery',on_event=journal.emit)
        (out/'execution.json').write_text(json.dumps(execution,indent=2))
        raw=probe.raw_path
        task.close_env();closed=True
        task.merge_pkl_to_hdf5_video()
        hdf5=out/'demonstration/data/episode0.hdf5'
        frames=raw.with_suffix('.frames.jsonl');obs=raw.with_suffix('.obs.jsonl')
        clock=audit_clock(execution['rows'],read_jsonl(raw)[1:],read_jsonl(frames)[1:],read_jsonl(obs)[1:],official_success=execution['official_success'])
        source=validate(raw,obs,frames,hdf5)
        require_attempt_identity(source,execution,seed)
        accepted=bool(execution['official_success'] and execution['termination']=='task_success'
                      and (condition=='expert_control' or execution['intervention_applied'])
                      and execution['physics_steps']<=60000 and task.plan_success)
        result=dict(schema='recovery_development_complete_v1',status='completed',seed=seed,condition=condition,
                    official_success=execution['official_success'],usable_recovery_candidate=accepted,
                    termination=execution['termination'],intervention_applied=execution['intervention_applied'],
                    action_count=execution['action_count'],physics_steps=execution['physics_steps'],
                    wall_seconds=execution['wall_seconds'],
                    source_hostname=socket.gethostname(),source_machine_role='relay_single_rtx3090',
                    source_audit=source,clock_audit=clock,protocol_sha256=sha256(PROTOCOL),
                    action_journal_sha256=sha256(out/'action_journal.jsonl'),training_approval=False,
                    limitation='Development candidate only; successful source audit does not authorize training or certify learned recovery.')
        (out/'result.json').write_text(json.dumps(result,indent=2))
    except Exception as exc:
        error=dict(status='needs_inspection',seed=seed,condition=condition,error_type=type(exc).__name__,
                   error=str(exc),physics_steps=getattr(probe,'step',None),training_approval=False)
        (out/'failure.json').write_text(json.dumps(error,indent=2));raise
    finally:
        if task is not None and not closed:
            try:task.close_env()
            except Exception as exc:
                (out/'cleanup_failure.json').write_text(json.dumps(dict(error=str(exc)),indent=2))
        if probe is not None:
            try:probe.close()
            except Exception:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=CONDITIONS,required=True);a=p.parse_args()
    run(a.seed,a.condition)

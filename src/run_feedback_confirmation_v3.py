"""In-process REAL tactile VLA + isolated RoboTwin, controlled-feedback confirmation candidate (admission manifest required).

No server capability is published. Every request carries current RGB/state and
causal live contact into the trained adapter. Original task success unchanged.
"""
import argparse,json,os,time,importlib
from pathlib import Path
import numpy as np
import torch
from probe_h50_training_backward import REPO,load_model
from probe_real_v2_bf16_stream import require_gpu_window,verify_fixed_inputs
from probe_real_v2_meta import build_old_config
from run_tactile_effect_pilot import infer,require_reproducibility_gate
from controlled_feedback_schedule import execution_count, matched_noise_seed
from controlled_feedback_timing import request_timing, execution_timing
from validate_expert_data import sha256

OVERLAY=REPO/'work/overlay_policy_equivalent_pilot_v1'


def make_transform(cfg):
    from types import SimpleNamespace
    import yaml
    from prepare_real_vla_visual_smoke import TRAIN,TOKENIZER,NORM,ROBOT
    from lingbotvla.models import build_processor
    from lingbotvla.data.vla_data.utils import FeatureTransform
    return FeatureTransform(str(ROBOT),SimpleNamespace(**yaml.safe_load(TRAIN.read_text())['data']),cfg,
        build_processor(str(TOKENIZER)),chunk_size=50,norm_stats_path=str(NORM))


def format_input(obs,transform):
    from prepare_real_vla_visual_smoke import CAMERAS
    from torchvision.transforms.v2 import Resize
    from h50_online_samples import INSTRUCTION
    formatted={name:obs['observation'][cam]['rgb'] for cam,name in CAMERAS.items()}
    formatted.update({'observation.state':obs['joint_action']['vector'],'task':INSTRUCTION})
    resize=Resize((256,256))
    x={name:resize(torch.as_tensor(formatted[name]).permute(2,0,1).contiguous().float()) for name in CAMERAS.values()}
    x.update({'observation.state':torch.as_tensor(formatted['observation.state']), 'task':INSTRUCTION})
    transformed=transform.apply(x,policy_eval=True)
    batch={k:transformed[k] if k=='image_grid_thw' else transformed[k].unsqueeze(0) for k in
        ('images','img_masks','state','lang_tokens','lang_masks','image_grid_thw')}
    return formatted,transformed,batch


def window_input(window):
    from tactile_rpc import encode_tactile_window
    p=encode_tactile_window(window)
    return {'tactile_history':torch.as_tensor(p['observation.tactile.history']).unsqueeze(0),
            'tactile_mask':torch.as_tensor(p['observation.tactile.mask']).unsqueeze(0),
            'tactile_age_ms':torch.tensor([float(p['observation.tactile.age_ms'])]),
            'tactile_valid':torch.tensor([bool(p['observation.tactile.valid'])])}


def run(seed,arm,repeat,steps,cadence,out):
    from verify_feedback_confirmation_v3 import verify
    confirmation_manifest = verify(seed, arm, repeat, cadence)
    require_reproducibility_gate()
    if seed not in range(100147,100163) or arm not in ('zero','true','mismatch','contact_state') or repeat not in (0,1,2) or steps != 2880 or cadence not in (1,10):raise ValueError('fixed paired protocol only')
    if Path.cwd().resolve()!=OVERLAY:raise ValueError('isolated RoboTwin cwd required')
    require_gpu_window();evidence,preview,manifest=verify_fixed_inputs(REPO)
    from patch_policy_equivalent_expert_overlay import generate
    from preregister_online_expert_trial import SOURCE,TASK_SOURCE,ROBOT_SOURCE,TASK_SOURCE_SHA,ROBOT_SOURCE_SHA
    import hashlib
    if (OVERLAY/'envs/_base_task.py').read_text()!=generate(SOURCE.read_text()):raise ValueError('overlay source drift')
    if sha256(TASK_SOURCE)!=TASK_SOURCE_SHA or sha256(ROBOT_SOURCE)!=ROBOT_SOURCE_SHA:raise ValueError('original simulator drift')
    from m7_tactile_controls import runtime_payload,WEIGHT_ARM
    from confirmation_tactile_controls import load_frozen_training_donors
    results=json.loads((REPO/f'data/reports/cwmj_m8_training_r{repeat}_{arm}/step_{steps}/result.json').read_text())
    pools,donor_manifest=load_frozen_training_donors(REPO) if arm=='mismatch' else ({},None)
    if donor_manifest is not None:
        if donor_manifest!=json.loads((REPO/'data/reports/cwmj_m6/confirmation_donors.json').read_text()):raise ValueError('runtime donor mapping drift')
        (out/'donor_manifest.json').write_text(json.dumps(donor_manifest,indent=2))
    meta=results;artifact=Path(meta['artifact'])
    if sha256(artifact)!=meta['sha256']:raise ValueError('trained adapter SHA mismatch')
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    apply_lingbot_qwen3_vl_patch();cfg=build_old_config();cfg.num_steps=10;cfg.use_cache=False
    transform=make_transform(cfg);model,_=load_model(cfg,preview,manifest)
    (out/'restored_buffers.json').write_text(json.dumps(model._nonpersistent_buffer_restore,indent=2))
    from safetensors.torch import load_file
    model.model.tactile_adapter.float().load_state_dict(load_file(str(artifact)),strict=True);model.eval()
    os.environ.update(FORCE_PROBE_EXPERT_DIR=str(out/'force'),TACTILE_LIVE_WINDOW='1',
        TACTILE_POLICY_EQUIVALENT_EXPERT='1',TACTILE_SOURCE_MACHINE_ROLE='relay_single_rtx3090')
    from record_online_expert_demo import _config
    from policy_rollout_capture import PolicyRolloutWriter
    from h50_online_samples import INSTRUCTION
    from cwmj_bottle_factory import install
    asset_evidence=install();asset_evidence['factory_sha256']=sha256(REPO/'src/cwmj_bottle_factory.py')
    (out/'scenario.json').write_text(json.dumps(asset_evidence,indent=2))
    task=importlib.import_module('envs.pick_diverse_bottles').pick_diverse_bottles()
    from feedback_topp_diagnostics import observed_action
    topp_events=[]
    writer=None;probe=None;started=time.monotonic();summary={};requests=[];applied=0
    try:
        env_cfg=_config(OVERLAY,out)
        env_cfg.update(eval_mode=True,save_data=True,policy_name='tactile_pilot',need_plan=True)
        task.setup_demo(now_ep_num=0,seed=seed,is_test=True,**env_cfg)
        task.set_instruction(INSTRUCTION);probe=task._force_probe
        if probe is None or probe.failed:raise RuntimeError('live probe unavailable')
        if probe.raw_path is not None:raise RuntimeError('policy clock must be declared before recording')
        probe.clock_origin='first_instrumented_policy_action_step_after_setup'
        initial={'bottle_ids':[int(task.bottle1_id),int(task.bottle2_id)],
            'bottle_points':[np.asarray(b.get_functional_point(0)).tolist() for b in (task.bottle1,task.bottle2)]}
        (out/'setup.json').write_text(json.dumps({'initial_scene':initial,'seed':seed,'arm':arm,'runner_sha256':sha256(Path(__file__))},indent=2))
        writer=PolicyRolloutWriter(out/'rollout.hdf5',task='pick_diverse_bottles',seed=seed,episode=0)
        reason='official_action_limit'
        while task.take_action_cnt<task.step_lim and not task.eval_success:
            if time.monotonic()-started>1200:reason='wall_time_truncation';break
            if probe.step>=200000:reason='physics_budget_truncation';break
            obs=task.get_obs();window=probe.policy_window();payload=window_input(window)
            formatted,transformed,batch=format_input(obs,transform)
            marker=probe.record_policy_request();writer.begin_request(marker,formatted,window)
            request_index=len(requests);t0=time.monotonic()
            timing=request_timing(seed=seed,executed_actions=applied,cadence=cadence,capture_step=probe.step,history_anchor=window['anchor_physics_step'],window_steps=window['window_steps'],dt_s=window['dt_s'])
            model_payload,donor=runtime_payload(payload,arm,applied,pools)
            prediction=infer(model,batch,model_payload,timing['noise_seed'])
            # Unnormalize original model state/action mapping, preserve 14D order.
            rev={k:v.clone() for k,v in transformed.items() if torch.is_tensor(v)}
            rev['actions']=prediction[0].float().cpu()
            commands=transform.unapply(rev)['action'].numpy()
            if commands.shape!=(50,14) or not np.isfinite(commands).all():raise RuntimeError('invalid decoded model command')
            request={'request_index':request_index,'physics_step':probe.step,'valid_tactile':bool(payload['tactile_valid'][0]),
                'past_contact':bool((payload['tactile_history'][...,:2]>0).any()),'infer_seconds':time.monotonic()-t0,
                'donor':donor,'input_history_sha256':hashlib.sha256(model_payload['tactile_history'].numpy().tobytes()).hexdigest()}
            request.update(timing);request['executed_action_timing']=[]
            requests.append(request)
            for command in commands[:execution_count(cadence,applied)]:
                if task.take_action_cnt>=task.step_lim or task.eval_success:break
                if probe.step>=200000 or time.monotonic()-started>1200:break
                before=probe.step
                observed_action(task,command,emit=topp_events.append,context=lambda:dict(
                    action_index=applied,request_index=request_index,physics_step=probe.step))
                writer.applied_action(command,physics_step_before=before,physics_step_after=probe.step);applied+=1
                request['executed_action_timing'].append(execution_timing(timing,physics_before=before,physics_after=probe.step,inference_seconds=request['infer_seconds']))
                if probe.failed:raise RuntimeError('contact probe failed')
            if request_index%5==0:print('ROLLOUT',seed,arm,request_index,applied,probe.step,flush=True)
        success=bool(task.eval_success and task.check_success())
        if time.monotonic()-started>1200:reason='wall_time_truncation'
        elif probe.step>200000:reason='physics_budget_truncation'
        elif success:reason='task_success'
        final_points=[np.asarray(b.get_functional_point(0)).tolist() for b in (task.bottle1,task.bottle2)]
        (out/'episode_outcome.json').write_text(json.dumps({'success':success,'termination':reason,'actions':applied,'physics_steps':probe.step,'initial_scene':initial,'final_bottle_points':final_points},indent=2))
        raw=probe.raw_path
        writer.close(observed_success=success);task.close_env();probe.close()
        from validate_policy_rollout import validate
        audit=validate(raw,raw.with_suffix('.obs.jsonl'),raw.with_suffix('.policy.jsonl'),out/'rollout.hdf5')
        from audit_controlled_feedback_timing import audit_hdf5
        censored=reason in ('wall_time_truncation','physics_budget_truncation')
        cadence_audit=(dict(actual_cadence_verified=False,reason='resource_censored',training_approval=False)
            if censored else audit_hdf5(out/'rollout.hdf5',requests,seed=seed,cadence=cadence,official_success=success))
        summary={'status':'completed','seed':seed,'arm':arm,'success':success,'termination':reason,'actions':applied,
            'physics_steps':probe.step,'requests':requests,'initial_scene':initial,'final_bottle_points':final_points,'cadence_audit':cadence_audit,
            'original_action_limit':task.step_lim,'replan_steps':cadence,'denoise_steps':10,'temporal_ensemble':False,
            'adapter_sha256':meta['sha256'],'repeat_index':repeat,'training_steps':steps,'weight_arm':WEIGHT_ARM[arm],'confirmation_manifest_sha256':sha256(REPO/'data/reports/feedback_confirmation_v3/manifest.json'),'confirmation_candidate':True,'training_approval':False,'resource_censored':reason in ('wall_time_truncation','physics_budget_truncation'),'source_evidence':evidence,'source_audit':audit,'wall_seconds':time.monotonic()-started}
    except Exception as exc:
        summary={'status':'failed','seed':seed,'arm':arm,'exception':type(exc).__name__,'error':str(exc),
                 'actions':applied,'physics_steps':getattr(probe,'step',0),'requests':requests,
                 'repeat_index':repeat,'replan_steps':cadence,'training_steps':steps,'training_approval':False}
        raise
    finally:
        if writer is not None:writer.close()
        try:task.close_env()
        except Exception:pass
        (out/'topp_events.json').write_text(json.dumps(topp_events,indent=2))
        (out/'result.json').write_text(json.dumps(summary,indent=2))
    print('CLOSED_LOOP_RESULT',json.dumps({k:v for k,v in summary.items() if k not in ('requests','source_audit')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=range(100147,100163),required=True)
    p.add_argument('--repeat',type=int,choices=(0,1,2),required=True)
    p.add_argument('--arm',choices=('zero','true','mismatch','contact_state'),required=True)
    p.add_argument('--cadence',type=int,choices=(1,10),required=True)
    a=p.parse_args()
    from verify_feedback_confirmation_v3 import verify
    verify(a.seed,a.arm,a.repeat,a.cadence)
    out=REPO/f'data/eval/feedback_confirmation_v3/r{a.repeat}_s{a.seed}_{a.arm}_c{a.cadence}'
    out.mkdir(parents=True,exist_ok=False)
    run(a.seed,a.arm,a.repeat,2880,a.cadence,out)

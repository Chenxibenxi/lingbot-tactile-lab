"""Replay M1 setup only, recording poses around the UNCHANGED stability test.
No policy/expert actions, no training labels, no retries promoted to datasets.
"""
import json,inspect,textwrap,types,re,hashlib,os
from pathlib import Path
import numpy as np
from probe_real_v2_bf16_stream import require_gpu_window
from record_online_expert_demo import _config
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]
OVERLAY=R/'work/overlay_policy_equivalent_pilot_v1'

def run():
    require_gpu_window()
    if Path.cwd().resolve()!=OVERLAY:raise ValueError('isolated original overlay required')
    for key in ['FORCE_PROBE_EXPERT_DIR','TACTILE_LIVE_WINDOW','TACTILE_POLICY_EQUIVALENT_EXPERT']:os.environ.pop(key,None)
    from envs.pick_diverse_bottles import pick_diverse_bottles
    from envs._base_task import Base_Task
    from cwmj_bottle_factory import install
    asset_evidence=install()
    original=Base_Task.check_stable
    source=textwrap.dedent(inspect.getsource(original))
    # Only observe immediately AFTER each original physics step; preserve loops,
    # quaternion metric, 3.0 threshold, examined actors and return exactly.
    source,n=re.subn(r'(?m)^(\s*)self.scene.step\(\)$',lambda m:m.group(0)+'\n'+m.group(1)+'self._m2_capture_stability()',source)
    if n!=2:raise ValueError('unexpected original stability source')
    scope=dict(original.__globals__);exec(compile(source,'<m2_observed_original_stability>','exec'),scope)
    root=R/'data/reports/cwmj_initialization_m3';root.mkdir(exist_ok=False)
    (root/'instrumented_check_stable.py').write_text(source)
    for seed in range(100033,100041):
        out=root/f'seed_{seed}';out.mkdir();task=pick_diverse_bottles();trace=[]
        def capture(self):
            trace.append({'step':len(trace)+1,'poses':[np.r_[b.get_pose().p,b.get_pose().q].tolist() for b in (self.bottle1,self.bottle2)]})
        task._m2_capture_stability=types.MethodType(capture,task)
        task.check_stable=types.MethodType(scope['check_stable'],task)
        result={'asset_replacement':asset_evidence,'seed':seed,'purpose':'paired_asset_setup_diagnostic_not_original_scenario_or_training_data','original_check_sha256':hashlib.sha256(textwrap.dedent(inspect.getsource(original)).encode()).hexdigest(),'instrumented_check_sha256':hashlib.sha256(source.encode()).hexdigest(),'training_labels':False}
        try:
            cfg=_config(OVERLAY,out);cfg.update(save_data=False,need_plan=False)
            task.setup_demo(now_ep_num=0,seed=seed,**cfg);result['setup']='accepted'
        except Exception as exc:result.update(setup='rejected',exception=type(exc).__name__,error=str(exc))
        try:
            result['bottle_ids']=[int(task.bottle1_id),int(task.bottle2_id)]
            result['assets']={'cwmj_converted_manifest':asset_evidence['manifest_sha256']}
            obs=task.get_obs()
            import cv2
            cv2.imwrite(str(out/'head.png'),cv2.cvtColor(obs['observation']['head_camera']['rgb'],cv2.COLOR_RGB2BGR))
        except Exception as exc:result['observation_error']=f'{type(exc).__name__}: {exc}'
        result['recorded_steps']=len(trace)
        (out/'trace.json').write_text(json.dumps(trace))
        (out/'result.json').write_text(json.dumps(result,indent=2))
        task.close_env();del task
        print('M2_SETUP',seed,result['setup'],result.get('bottle_ids'),len(trace),flush=True)
if __name__=='__main__':run()

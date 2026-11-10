"""Admission for input-only comparisons under one fixed true-trained adapter."""
import json,shutil
from pathlib import Path
from validate_expert_data import sha256
from verify_controlled_feedback_pilot_v2 import verify as verify_original
R=Path(__file__).resolve().parents[1]
def verify(seed,arm,cadence):
    if arm not in ('live','history_zero','history_mismatch','contact_bit'):raise ValueError('fixed input-only condition required')
    original=verify_original(seed,'true',cadence)
    p=R/'data/reports/fixed_weight_controls_v1/manifest.json';v=json.loads(p.read_text())
    if v.get('gpu_execution_ready') is not True:raise ValueError('not frozen')
    required={'src/run_fixed_weight_controls.py','src/verify_fixed_weight_controls.py','src/fixed_weight_tactile_control.py','docs/FIXED_WEIGHT_CONTROLS_PROTOCOL_V1_ZH.md'}
    if not required.issubset(v['files']):raise ValueError('missing source pins')
    for name,digest in v['files'].items():
        if sha256(R/name)!=digest:raise ValueError('source drift: '+name)
    source=R/'data/reports/cwmj_m8_training_r0_true/step_2880/result.json'
    meta=json.loads(source.read_text())
    if sha256(source)!=v['training_result_sha256'] or sha256(Path(meta['artifact']))!=v['adapter_sha256']:raise ValueError('weight drift')
    target=R/'data/eval/fixed_weight_controls_v1'
    if target.resolve()!=Path('/home/admin001/lingbot-tactile-storage/fixed-weight-controls-v1') or shutil.disk_usage(target).free<8*1024**3:raise ValueError('output volume/capacity')
    return original
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',required=True);p.add_argument('--cadence',type=int,required=True)
    a=p.parse_args();verify(a.seed,a.arm,a.cadence);print('Fixed-weight input protocol verified')

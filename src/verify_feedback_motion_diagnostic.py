"""Fail closed until the independent diagnostic implementation is frozen."""
import json
from pathlib import Path
from validate_expert_data import sha256
from verify_controlled_feedback_pilot_v2 import verify as verify_original
R=Path(__file__).resolve().parents[1]
def verify(seed,arm,cadence):
    if arm not in ('zero','true'):raise ValueError('diagnostic arms only')
    original=verify_original(seed,arm,cadence)
    manifest=json.loads((R/'data/reports/feedback_motion_diagnostic_run_v1/manifest.json').read_text())
    if manifest.get('gpu_execution_ready') is not True:raise ValueError('diagnostic not admitted')
    if manifest.get('stage')=='first integration trial only' and (seed,arm,cadence)!=(100145,'zero',10):raise ValueError('first integration trial must pass before further admission')
    required={'src/run_feedback_motion_diagnostic.py','src/verify_feedback_motion_diagnostic.py','src/feedback_prediction_trace.py','src/feedback_physical_state.py','docs/FEEDBACK_MOTION_DIAGNOSTIC_PROTOCOL_V1_ZH.md'}
    if not required.issubset(manifest['files']):raise ValueError('missing diagnostic pins')
    for path,digest in manifest['files'].items():
        if sha256(R/path)!=digest:raise ValueError('diagnostic source drift: '+path)
    if Path(manifest['original_manifest']).resolve()!=R/'data/reports/controlled_feedback_pilot_v2/manifest.json':raise ValueError('wrong original freeze')
    if sha256(Path(manifest['original_manifest']))!=manifest['original_manifest_sha256']:raise ValueError('original freeze changed')
    return original

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',required=True);p.add_argument('--cadence',type=int,required=True)
    a=p.parse_args();verify(a.seed,a.arm,a.cadence);print('Diagnostic freeze verified')

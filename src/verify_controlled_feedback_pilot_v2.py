"""Pilot admission checks; absent manifests intentionally prevent execution."""
import json
import shutil
from pathlib import Path
from validate_expert_data import sha256

R=Path(__file__).resolve().parents[1]


def verify(seed,arm,cadence):
    if (type(seed) is not int or seed not in (100145,100146)
            or arm not in ('zero','true','mismatch','contact_state')
            or type(cadence) is not int or cadence not in (1,10)):
        raise ValueError('fixed technical pilot matrix required')
    reservation=json.loads((R/'datasets/controlled_feedback_pilot_reservations_v2.json').read_text())
    protocol=R/'docs/CONTROLLED_FEEDBACK_PILOT_PROTOCOL_V2_ZH.md'
    if reservation['protocol_sha256']!=sha256(protocol):raise ValueError('pilot protocol changed')
    manifest=json.loads((R/'data/reports/controlled_feedback_pilot_v2/manifest.json').read_text())
    if manifest.get('gpu_execution_ready') is not True or manifest.get('protocol_sha256')!=sha256(protocol):
        raise ValueError('pilot implementation not frozen')
    required={'src/run_controlled_feedback_pilot_v2.py','src/verify_controlled_feedback_pilot_v2.py','src/feedback_topp_diagnostics.py',
              'src/controlled_feedback_schedule.py','src/controlled_feedback_timing.py',
              'src/audit_controlled_feedback_timing.py'}
    if not required.issubset(manifest['files']):raise ValueError('missing pilot source pins')
    for name,digest in manifest['files'].items():
        if sha256(R/name)!=digest:raise ValueError('pilot source drift: '+name)
    for name,digest in manifest['artifacts'].items():
        if sha256(Path(name))!=digest:raise ValueError('frozen adapter drift')
    if manifest.get('repeat')!=0 or manifest.get('training_steps')!=2880:
        raise ValueError('fixed r0/2880 adapters required')
    output=R/'data/eval/controlled_feedback_pilot_v2'
    if output.resolve()!=Path('/home/admin001/lingbot-tactile-storage/controlled-feedback-pilot-v2'):
        raise ValueError('registered local output volume required')
    if shutil.disk_usage(output).free<8*1024**3:
        raise ValueError('at least 8 GiB free on output volume required')
    return manifest

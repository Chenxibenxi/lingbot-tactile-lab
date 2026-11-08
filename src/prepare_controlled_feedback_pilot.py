"""CPU-only freeze for the fixed 16-episode feedback technical pilot."""
import json
from pathlib import Path
import subprocess
import sys
from verify_controlled_feedback_pilot import R
from validate_expert_data import sha256
from run_controlled_feedback_batch import write_new


def prepare():
    reservation=R/'datasets/controlled_feedback_pilot_reservations_v1.json'
    data=json.loads(reservation.read_text());protocol=R/'docs/CONTROLLED_FEEDBACK_PILOT_PROTOCOL_V1_ZH.md'
    if data['protocol_sha256']!=sha256(protocol) or data['seeds']!=[100127,100128]:
        raise ValueError('pilot reservation drift')
    output=R/'data/eval/controlled_feedback_pilot_v1'
    if list(output.iterdir()):raise ValueError('pilot storage must be pristine before freeze')
    for seed in data['seeds']:
        for folder in ('data/paired','data/eval'):
            if list((R/folder).glob(f'*{seed}*')):raise ValueError('pilot seed already used')
    from verify_cwmj_m8_confirmation import verify_confirmation
    names=set();artifacts={}
    for arm in data['arms']:
        previous=verify_confirmation(R,100103,0,arm,2880)
        names.update(previous['code_sha256']);names.update(previous['evidence_sha256'])
        node=R/f'data/reports/cwmj_m8_training_r0_{arm}/step_2880/result.json'
        meta=json.loads(node.read_text());weight=previous['weights']['0'][arm]['2880']
        if any(meta[k]!=weight[k] for k in ('artifact','sha256')):raise ValueError('node metadata drift')
        artifacts[meta['artifact']]=meta['sha256'];names.add(str(node.relative_to(R)))
    for old in ('data/reports/cwmj_m8/training_manifest.json','data/reports/recovery_development_v2/manifest.json'):
        previous=json.loads((R/old).read_text())
        for name,digest in previous['files'].items():
            if sha256(R/name)!=digest:raise ValueError('prior source drift: '+name)
            names.add(name)
    names.update(str(p.relative_to(R)) for p in (R/'src').glob('*controlled_feedback*.py'))
    names.update(['scripts/run_controlled_feedback_pilot.sh',str(protocol.relative_to(R)),str(reservation.relative_to(R)),
        'src/policy_rollout_capture.py','src/validate_policy_rollout.py','src/validate_policy_trace.py',
        'src/confirmation_tactile_controls.py','src/force_probe.py','src/tactile_rpc.py',
        'src/live_tactile_window.py','src/window_math.py','src/align_temporal.py',
        'data/reports/cwmj_m6/confirmation_donors.json','data/reports/cwmj_m6/training_manifest.json'])
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(R/'tests'),'-p','test_controlled_feedback*.py'],check=True)
    subprocess.run(['bash','-n',str(R/'scripts/run_controlled_feedback_pilot.sh')],check=True)
    manifest=dict(schema='controlled_feedback_pilot_v1',gpu_execution_ready=True,protocol_sha256=sha256(protocol),
        seeds=data['seeds'],arms=data['arms'],cadences=[10,1],repeat=0,training_steps=2880,
        artifacts=artifacts,files={name:sha256(R/name) for name in sorted(names)},
        tests={str(p.relative_to(R)):sha256(p) for p in sorted((R/'tests').glob('test_controlled_feedback*.py'))},
        confirmation_eligible=False,training_approval=False)
    root=R/'data/reports/controlled_feedback_pilot_v1';root.mkdir(exist_ok=False)
    write_new(root/'manifest.json',manifest)
    print(root/'manifest.json')


if __name__=='__main__':prepare()

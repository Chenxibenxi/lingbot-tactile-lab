"""Deny execution until independent pilot audit and formal source freeze exist."""
import json
import shutil
from pathlib import Path
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]
SEEDS=list(range(100129,100145))
ARMS=('zero','true','mismatch','contact_state')


def identity(seed,arm,repeat,cadence):
    if type(seed) is not int or seed not in SEEDS or arm not in ARMS or type(repeat) is not int or repeat not in (0,1,2) or type(cadence) is not int or cadence not in (1,10):
        raise ValueError('registered seed/repeat/condition/frequency required')
    return f'r{repeat}_s{seed}_{arm}_c{cadence}'


def verify(seed,arm,repeat,cadence,root=R):
    identity(seed,arm,repeat,cadence);root=Path(root)
    manifest=json.loads((root/'data/reports/feedback_confirmation_v1/manifest.json').read_text())
    if manifest.get('gpu_execution_ready') is not True or manifest.get('schema')!='feedback_confirmation_frozen_v1':
        raise ValueError('formal confirmation not admitted')
    if manifest.get('seeds')!=SEEDS or manifest.get('repeats')!=[0,1,2] or manifest.get('training_steps')!=2880:
        raise ValueError('confirmation matrix drift')
    required={'src/run_feedback_confirmation.py','src/verify_feedback_confirmation.py',
              'src/feedback_confirmation_statistics.py','src/feedback_outcome_contract.py',
              'src/audit_feedback_confirmation.py','src/report_feedback_confirmation_final.py',
              'src/report_feedback_confirmation_episode.py','src/run_feedback_confirmation_batch.py',
              'scripts/run_feedback_confirmation.sh','src/controlled_feedback_schedule.py',
              'src/controlled_feedback_timing.py','src/audit_controlled_feedback_timing.py',
              'docs/CONTROLLED_FEEDBACK_CONFIRMATION_PROTOCOL_V1_ZH.md',
              'data/reports/controlled_feedback_pilot_v1_final/summary.json'}
    if not required.issubset(manifest['files']):raise ValueError('missing protocol/audit/code pins')
    for name,digest in manifest['files'].items():
        if sha256(root/name)!=digest:raise ValueError('confirmation source drift: '+name)
    pilot=json.loads((root/'data/reports/controlled_feedback_pilot_v1_final/summary.json').read_text())
    if pilot.get('technical_gate') is not True:raise ValueError('pilot technical gate failed')
    expected={f'r{r}_{a}' for r in (0,1,2) for a in ARMS}
    if set(manifest['adapters'])!=expected:raise ValueError('all twelve adapter identities required')
    for r in (0,1,2):
        for a in ARMS:
            record=manifest['adapters'][f'r{r}_{a}']
            source=root/f'data/reports/cwmj_m8_training_r{r}_{a}/step_2880/result.json'
            if sha256(source)!=record['result_sha256']:raise ValueError('training result drift')
            value=json.loads(source.read_text())
            if value['artifact']!=record['artifact'] or value['sha256']!=record['sha256'] or sha256(Path(record['artifact']))!=record['sha256']:
                raise ValueError('adapter identity/bytes mismatch')
    output=root/'data/eval/feedback_confirmation_v1'
    if output.resolve()!=Path('/home/admin001/lingbot-tactile-storage/feedback-confirmation-v1'):
        raise ValueError('separate registered output volume required')
    minimum=manifest.get('minimum_free_output_bytes')
    if type(minimum) is not int or minimum<8*1024**3 or shutil.disk_usage(output).free<minimum:
        raise ValueError('frozen capacity budget not met')
    return manifest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--repeat',type=int,required=True);p.add_argument('--arm',required=True)
    p.add_argument('--cadence',type=int,required=True);a=p.parse_args()
    verify(a.seed,a.arm,a.repeat,a.cadence)
    print('Formal feedback admission verified')

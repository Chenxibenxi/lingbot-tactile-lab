"""CPU-only freeze of the reserved recovery development implementation."""
import json
from pathlib import Path
import subprocess
import sys
from validate_expert_data import sha256
from verify_m7_training import verify

R=Path(__file__).resolve().parents[1]
FILES=[
    'docs/RECOVERY_DEVELOPMENT_PROTOCOL_V1_ZH.md','datasets/recovery_development_reservations_v1.json',
    'src/record_recovery_development.py','src/recovery_expert_executor.py','src/recovery_action_journal.py',
    'src/recovery_clock_audit.py','src/recovery_supervision_contract.py','src/audit_recovery_pair.py',
    'src/report_recovery_development.py','src/report_recovery_development_batch.py',
    'src/online_bottle_expert_actions.py','src/online_expert_pilot_executor.py',
    'src/record_online_expert_demo.py','src/validate_expert_data.py','src/cwmj_bottle_factory.py',
    'src/patch_policy_equivalent_expert_overlay.py','src/make_expert_overlay.py','src/make_overlay.py',
    'src/preregister_online_expert_trial.py','src/probe_real_v2_bf16_stream.py',
    'work/cwmj_bottle_asset_v2/manifest.json','work/run_recovery_development.sh',
    'work/run_recovery_development_batch.py','src/prepare_recovery_development.py']


def prepare():
    verify(R)
    reservation=json.loads((R/'datasets/recovery_development_reservations_v1.json').read_text())
    protocol=R/'docs/RECOVERY_DEVELOPMENT_PROTOCOL_V1_ZH.md'
    if reservation['protocol_sha256']!=sha256(protocol):raise ValueError('reserved protocol changed')
    if [x['seed'] for x in reservation['seeds']]!=list(range(100119,100123)):raise ValueError('development matrix changed')
    for seed in range(100119,100123):
        for folder in ('data/paired','data/eval'):
            if list((R/folder).glob(f'*{seed}*')):raise ValueError('development seed already consumed')
    tests=list((R/'tests').glob('test_recovery*.py'))+[R/'tests/test_report_recovery_development.py']
    for pattern in ('test_recovery*.py','test_report_recovery_development.py'):
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(R/'tests'),'-p',pattern],check=True)
    subprocess.run(['bash','-n',str(R/'work/run_recovery_development.sh')],check=True)
    asset=R/'work/cwmj_bottle_asset_v2';asset_manifest=json.loads((asset/'manifest.json').read_text())
    for name,digest in asset_manifest['converted_files_sha256'].items():
        if sha256(asset/name)!=digest:raise ValueError('converted asset changed')
    for name,digest in asset_manifest['source_files_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('original asset changed')
    root=R/'data/reports/recovery_development_v1';root.mkdir(exist_ok=False)
    manifest=dict(schema='recovery_development_implementation_v1',gpu_execution_ready=True,
                  authorization='user authorized autonomous local simulation; run only after M8 completes',
                  protocol_sha256=sha256(protocol),seeds=list(range(100119,100123)),
                  conditions=['expert_control','left_open_once_recovery'],
                  files={n:sha256(R/n) for n in FILES},tests={str(p.relative_to(R)):sha256(p) for p in tests},
                  tests_passed=True,training_approval=False)
    with (root/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(root/'manifest.json')


if __name__=='__main__':prepare()

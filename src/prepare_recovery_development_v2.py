"""Explicit CPU-only readiness freeze; no GPU execution."""
import json
from pathlib import Path
import subprocess
import sys
from record_recovery_development_v2 import R, PROTOCOL, KINDS, write_new
from validate_expert_data import sha256


def prepare():
    reservation = json.loads((R/'datasets/recovery_development_reservations_v2.json').read_text())
    if reservation['protocol_sha256'] != sha256(PROTOCOL): raise ValueError('protocol changed')
    if [x['seed'] for x in reservation['seeds']] != list(range(100123,100127)):
        raise ValueError('fixed seeds changed')
    for seed in range(100123,100127):
        for folder in ('data/paired','data/eval'):
            if list((R/folder).glob(f'*{seed}*')): raise ValueError('seed already consumed')
    # Carry over every original dependency pin; do not silently accept baseline changes.
    previous = json.loads((R/'data/reports/recovery_development_v1/manifest.json').read_text())
    for name,digest in previous['files'].items():
        if sha256(R/name) != digest: raise ValueError('previous immutable dependency drift: '+name)
    names = set(previous['files'])
    names.update(str(p.relative_to(R)) for p in (R/'src').glob('*recovery*v2.py'))
    names.update(['scripts/run_recovery_development_v2.sh',
                  'docs/RECOVERY_DEVELOPMENT_PROTOCOL_V2_ZH.md','datasets/recovery_development_reservations_v2.json'])
    for pattern in ('test_recovery*.py',):
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(R/'tests'),'-p',pattern],check=True)
    subprocess.run(['bash','-n',str(R/'scripts/run_recovery_development_v2.sh')],check=True)
    asset = R/'work/cwmj_bottle_asset_v2'
    assets = json.loads((asset/'manifest.json').read_text())
    for name,digest in assets['converted_files_sha256'].items():
        if sha256(asset/name) != digest: raise ValueError('converted asset drift')
    for name,digest in assets['source_files_sha256'].items():
        if sha256(Path(name)) != digest: raise ValueError('original asset drift')
    manifest = dict(schema='recovery_development_implementation_v2',gpu_execution_ready=True,
        protocol_sha256=sha256(PROTOCOL),seeds=list(range(100123,100127)),kinds=list(KINDS),
        files={name:sha256(R/name) for name in sorted(names)},
        tests={str(p.relative_to(R)):sha256(p) for p in sorted((R/'tests').glob('test_recovery*.py'))},
        tests_passed=True,training_approval=False,
        limitation='Readiness only, not evidence of successful simulation or data admission.')
    root = R/'data/reports/recovery_development_v2'; root.mkdir(exist_ok=False)
    write_new(root/'manifest.json',manifest)
    print(root/'manifest.json')


if __name__ == '__main__': prepare()

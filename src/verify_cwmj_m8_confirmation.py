"""Fail closed until completed training and the full M8 evaluation are pinned."""
import json
from pathlib import Path
from validate_expert_data import sha256
from verify_m7_training import verify
from m8_paired_seed_statistics import CONDITIONS, STEPS

FILES = ['src/run_cwmj_m8_confirmation.py', 'src/verify_cwmj_m8_confirmation.py',
         'src/m8_paired_seed_statistics.py', 'src/m7_paired_seed_statistics.py',
         'src/m7_tactile_controls.py', 'src/report_cwmj_m8_confirmation.py',
         'src/audit_cwmj_m8_pair.py', 'src/report_cwmj_m8_confirmation_batch.py',
         'src/audit_cwmj_m7_pair.py', 'src/m8_completion_inventory.py',
         'src/queue_strata_m8_analysis.py', 'src/strata_research_analysis.py',
         'src/prepare_cwmj_m8_confirmation.py',
         'work/run_cwmj_m8_confirmation.sh', 'work/run_cwmj_m8_confirmation_batch.py']


def verify_confirmation(repo, seed, repeat, condition, steps):
    if (type(seed) is not int or seed not in range(100103,100119)
            or type(repeat) is not int or repeat not in range(3)
            or condition not in CONDITIONS or type(steps) is not int or steps not in STEPS):
        raise ValueError('fixed M8 confirmation matrix required')
    verify(repo)
    training=json.loads((repo/'data/reports/cwmj_m8/training_manifest.json').read_text())
    if training.get('frozen') is not True:raise ValueError('training not frozen')
    for name,digest in training['files'].items():
        if sha256(repo/name)!=digest:raise ValueError('M8 training provenance changed: '+name)
    manifest=json.loads((repo/'data/reports/cwmj_m8/confirmation_manifest.json').read_text())
    if (manifest['seeds']!=list(range(100103,100119)) or manifest['conditions']!=list(CONDITIONS)
            or manifest['steps']!=list(STEPS) or manifest['repetitions']!=[0,1,2]):
        raise ValueError('M8 matrix changed')
    if set(manifest['code_sha256'])!=set(FILES):raise ValueError('incomplete evaluation code pins')
    for name,digest in manifest['code_sha256'].items():
        if sha256(repo/name)!=digest:raise ValueError('evaluation source changed: '+name)
    required=['data/reports/cwmj_m8/training_manifest.json','docs/CWMJ_M8_FROZEN_PROTOCOL_ZH.md',
              'datasets/tactile_seed_bank_m8_reservations_v1.json']
    required += [f'data/reports/cwmj_m8_training_r{r}_{c}/result.json' for r in range(3) for c in CONDITIONS]
    if set(manifest['evidence_sha256'])!=set(required):raise ValueError('incomplete evidence pins')
    for name,digest in manifest['evidence_sha256'].items():
        if sha256(repo/name)!=digest:raise ValueError('training evidence changed: '+name)
    weight=manifest['weights'][str(repeat)][condition][str(steps)]
    folder=repo/f'data/reports/cwmj_m8_training_r{repeat}_{condition}'
    result=json.loads((folder/'result.json').read_text())
    if result['status']!='completed' or [n['steps'] for n in result['nodes']]!=[720,1440,2880]:
        raise ValueError('all fixed training nodes required')
    node=next(n for n in result['nodes'] if n['steps']==steps)
    if not node['historical_720_exact'] or node['changed_tensors']<1:
        raise ValueError('historical reproducibility/optimizer gate failed')
    if weight!={k:node[k] for k in ('artifact','sha256')} or sha256(Path(weight['artifact']))!=weight['sha256']:
        raise ValueError('evaluation weight changed')
    if list((repo/'data/paired').glob(f'*{seed}*')):
        raise ValueError('test seed used for collection')
    if any(not p.name.startswith('cwmj_m8_confirm_') for p in (repo/'data/eval').glob(f'*{seed}*')):
        raise ValueError('test seed used outside fixed experiment')
    return manifest

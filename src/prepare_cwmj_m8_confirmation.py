"""Freeze all fixed duration weights only after the entire training matrix."""
import json
from pathlib import Path
from validate_expert_data import sha256
from verify_m7_training import verify
from verify_cwmj_m8_confirmation import FILES,verify_confirmation
from m8_paired_seed_statistics import CONDITIONS,STEPS

R=Path(__file__).resolve().parents[1]


def prepare():
    verify(R)
    code=json.loads((R/'data/reports/cwmj_m8/evaluation_code_lock.json').read_text())
    if set(code['files'])!=set(FILES+['work/continue_m8_after_training.py']):raise ValueError('incomplete preflight code lock')
    for p,h in code['files'].items():
        if sha256(R/p)!=h:raise ValueError('evaluation code changed after preflight')
    training=R/'data/reports/cwmj_m8/training_manifest.json'
    for p,h in json.loads(training.read_text())['files'].items():
        if sha256(R/p)!=h:raise ValueError('training provenance changed')
    weights={};evidence=['data/reports/cwmj_m8/training_manifest.json','docs/CWMJ_M8_FROZEN_PROTOCOL_ZH.md','datasets/tactile_seed_bank_m8_reservations_v1.json']
    for r in range(3):
        weights[str(r)]={}
        for arm in CONDITIONS:
            name=f'data/reports/cwmj_m8_training_r{r}_{arm}/result.json';result=json.loads((R/name).read_text())
            if result['status']!='completed' or [n['steps'] for n in result['nodes']]!=[720,1440,2880]:raise ValueError('incomplete training')
            if result['protocol']['manifest_sha256']!=sha256(training):raise ValueError('wrong training manifest')
            weights[str(r)][arm]={}
            for node in result['nodes']:
                if not node['historical_720_exact'] or node['arm']!=arm or node['repeat']!=r or node['changed_tensors']<1:raise ValueError('invalid training node')
                if sha256(Path(node['artifact']))!=node['sha256']:raise ValueError('weight changed')
                if node['steps'] in STEPS:weights[str(r)][arm][str(node['steps'])]={k:node[k] for k in ('artifact','sha256')}
            evidence.append(name)
    for seed in range(100103,100119):
        if any(list((R/folder).glob(f'*{seed}*')) for folder in ('data/paired','data/eval')):raise ValueError('test seed already used')
    manifest=dict(seeds=list(range(100103,100119)),conditions=list(CONDITIONS),steps=list(STEPS),repetitions=[0,1,2],weights=weights,
                  code_sha256={n:sha256(R/n) for n in FILES},evidence_sha256={n:sha256(R/n) for n in evidence})
    path=R/'data/reports/cwmj_m8/confirmation_manifest.json'
    with path.open('x') as f:json.dump(manifest,f,indent=2)
    verify_confirmation(R,100103,0,'zero',720)


if __name__=='__main__':prepare()

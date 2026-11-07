"""Freeze all qualified fixed-seed CWMJ episodes; no result-driven model selection."""
import json
from pathlib import Path
from prepare_cwmj_m4 import FOLDS,REPO
from cwmj_m4_samples import M4TerminalEpisode
from validate_expert_data import sha256

def prepare():
    root=REPO/'data/reports/cwmj_m4';train=[];val=[];audits=[];rejected=[]
    for seed,fold in FOLDS.items():
        report=root/f'seed_{seed}/result.json'
        r=json.loads(report.read_text())
        if r['status']!='accepted_success_source_verified':rejected.append({'seed':seed,'fold':fold,'status':r['status'],'failure':r['failure']});continue
        ep=M4TerminalEpisode(seed);samples=[ep.sample(i,False) for i in range(len(ep))];n=sum(s['metadata']['contact'] for s in samples)
        if n<4 or len(samples)-n<4:rejected.append({'seed':seed,'fold':fold,'status':'insufficient_predeclared_contact_strata'});continue
        (train if fold=='train' else val).append(seed)
        audits.append({'seed':seed,'fold':fold,'samples':len(samples),'contact_samples':n,'source_sha256':ep.hashes})
    gate={'train_seeds':train,'validation_seeds':val,'sources':audits,'rejected':rejected,'passed':len(train)>=2 and len(val)>=1,
          'scenario':'cwmj_single_wine_bottle_original_task_predicate','original_scenario_mixed':False,
          'collection_protocol_sha256':sha256(REPO/'docs/CWMJ_M4_COLLECTION_PROTOCOL.md'),
          'training_protocol_sha256':sha256(REPO/'docs/CWMJ_M4_TRAINING_PROTOCOL_ZH.md')}
    path=root/('training_manifest.json' if gate['passed'] else 'data_gate_failed.json')
    with path.open('x') as f:json.dump(gate,f,indent=2)
    print(json.dumps(gate,indent=2));return gate
if __name__=='__main__':prepare()

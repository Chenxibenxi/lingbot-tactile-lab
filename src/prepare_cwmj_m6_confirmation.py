"""Freeze all nine final models and runtime donors, never rank checkpoints."""
import json
from pathlib import Path
from validate_expert_data import sha256
from verify_cwmj_m6_confirmation import CODE_FILES
from confirmation_tactile_controls import CONDITIONS,load_frozen_training_donors
R=Path(__file__).resolve().parents[1]
def prepare():
    root=R/'data/reports/cwmj_m6';summary_path=R/'data/reports/cwmj_m6_training_summary.json';summary=json.loads(summary_path.read_text())
    if summary['repeats']!=3 or not summary['distinct_initializations'] or not summary['frozen_validation_exact_equal']:raise ValueError('all repeated training and infrastructure gates must complete')
    weights={}
    for repeat in range(3):
        result=json.loads((R/f'data/reports/cwmj_m6_training_r{repeat}/result.json').read_text())
        weights[str(repeat)]={}
        for arm in ['zero','true','mismatch']:
            w=result['updates'][arm]
            if w['steps']!=720 or w['changed_tensors']<=0 or sha256(Path(w['artifact']))!=w['sha256']:raise ValueError('all fixed final models required')
            weights[str(repeat)][arm]={'artifact':w['artifact'],'sha256':w['sha256'],'steps':720}
    _,donors=load_frozen_training_donors(R)
    with (root/'confirmation_donors.json').open('x') as f:json.dump(donors,f,indent=2)
    m={'schema':'cwmj_m6_independent_confirmation_v1','seeds':list(range(100071,100087)),'repetitions':[0,1,2],'conditions':list(CONDITIONS),'planned_episodes':192,'weights':weights,'protocol_sha256':sha256(R/'docs/CWMJ_M6_TRAINING_PROTOCOL_ZH.md'),'training_manifest_sha256':sha256(root/'training_manifest.json'),'training_summary_sha256':sha256(summary_path),'donor_manifest_sha256':sha256(root/'confirmation_donors.json'),'code_sha256':{p:sha256(R/p) for p in CODE_FILES},'offline_gate_does_not_select_models':True}
    with (root/'confirmation_manifest.json').open('x') as f:json.dump(m,f,indent=2)
    print('CONFIRMATION_FROZEN',192)
if __name__=='__main__':prepare()

"""Read-only evidence -> separate versioned seed bank; never promote raw data."""
from pathlib import Path
import json
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]

def build():
    entries=[]
    manifest=json.loads((R/'data/reports/cwmj_m4/training_manifest.json').read_text())
    for scenario,batch,seeds,prefix in [('original_20_bottles','multiseed_m1',range(100033,100041),'relay_pickdiverse'),('cwmj_wine_v2','cwmj_m4',range(100045,100051),'relay_pickcwmj')]:
        for seed in seeds:
            report=R/f'data/reports/{batch}/seed_{seed}/result.json';row=json.loads(report.read_text())
            success=row['status']=='accepted_success_source_verified'
            fitted=scenario=='cwmj_wine_v2' and seed in manifest['train_seeds']
            entry={'scenario':scenario,'seed':seed,'role':'expert_collection_attempt','fold':row['fold'],'status':row['status'],
                'report':str(report.parent/'report.md'),'report_result_sha256':sha256(report),'valid_windows':row['valid_windows'],
                'contact_windows':row['contact_windows'],'completed_source_verified':success,
                'used_as_expert_supervision_in_m4':fitted,'future_fit_requires_new_manifest':True,
                'exception':(row.get('failure') or {}).get('exception_type'),'preflight_reason':row.get('preflight_reason'),
                'raw_dir':str(R/f'data/paired/{prefix}_s{seed}_online_h50_stride20_{row["fold"]}_expert0')}
            if success:entry['source_sha256']=row['source_audit']['source_file_sha256']
            entries.append(entry)
    for seed in range(100051,100055):
        for arm in ['zero','true']:
            trial=R/f'data/eval/cwmj_m4_{seed}_{arm}';result=trial/'result.json'
            e={'scenario':'cwmj_wine_v2','seed':seed,'arm':arm,'role':'heldout_policy_rollout','fold':'test',
               'status':'running_or_incomplete' if trial.exists() else 'reserved','raw_dir':str(trial),
               'used_as_expert_supervision_in_m4':False,'future_fit_requires_new_manifest':True}
            if result.exists():
                r=json.loads(result.read_text());e.update(status=r['status'],success=r.get('success'),termination=r.get('termination',r.get('exception')),result_sha256=sha256(result),actions=r['actions'])
            entries.append(e)
    for scenario in {e['scenario'] for e in entries}:
        folds={f:{e['seed'] for e in entries if e['scenario']==scenario and e['fold']==f} for f in ['train','validation','test']}
        if folds['train']&folds['validation'] or folds['train']&folds['test'] or folds['validation']&folds['test']:raise ValueError('seed split leakage')
    return {'schema':'lingbot_tactile_seed_bank_v1','scope':'M1-M4 relay evidence only; historical competition not relabeled',
            'original_registry_unchanged':True,'diagnostic_replays_are_not_independent_samples':True,
            'scenario_definitions':{'original_20_bottles':'original RoboTwin 001_bottle random IDs0–19','cwmj_wine_v2':{'asset_manifest':str(R/'work/cwmj_bottle_asset_v2/manifest.json'),'sha256':sha256(R/'work/cwmj_bottle_asset_v2/manifest.json')}},
            'm4_training_manifest_sha256':sha256(R/'data/reports/cwmj_m4/training_manifest.json'),'entries':entries}
if __name__=='__main__':
    bank=build();out=R/'datasets/tactile_seed_bank_m1_m4_v1.json';out.write_text(json.dumps(bank,indent=2));print('SEED_BANK_ENTRIES',len(bank['entries']))

"""Extend immutable evidence roles; reused test runs never count as new seeds."""
from pathlib import Path
import json
from build_tactile_seed_bank import build,R
from validate_expert_data import sha256

def main():
 bank=build();bank['schema']='lingbot_tactile_seed_bank_v2';bank['scope']='M1-M6 relay evidence, with M5 reused-test interventions explicit'
 for seed in range(100051,100055):
  trial=R/f'data/eval/cwmj_m5_{seed}_zero';p=trial/'result.json'
  e={'scenario':'cwmj_wine_v2','seed':seed,'fold':'test','stage':'M5','role':'exploratory_same_weight_input_ablation','independent_new_seed':False,'weight_arm':'true','input_arm':'zero','raw_dir':str(trial),'status':'running_or_incomplete' if trial.exists() else 'reserved','used_as_expert_supervision_in_m4':False,'future_fit_requires_new_manifest':True}
  if p.exists():
   r=json.loads(p.read_text());e.update(status=r['status'],success=r.get('success'),termination=r.get('termination',r.get('exception')),actions=r['actions'],result_sha256=sha256(p))
  bank['entries'].append(e)
 for seed in range(100055,100071):
  fold='train' if seed<100067 else 'validation';trial=R/f'data/paired/relay_pickcwmjm6_s{seed}_online_h50_stride20_{fold}_expert0';p=R/f'data/reports/cwmj_m6/seed_{seed}/result.json'
  e={'scenario':'cwmj_wine_v2','seed':seed,'fold':fold,'stage':'M6','role':'expert_collection_attempt','raw_dir':str(trial),'status':'running_or_incomplete' if trial.exists() else 'reserved','used_as_expert_supervision_in_m4':False,'used_as_expert_supervision_in_m6':False,'future_fit_requires_new_manifest':True}
  if p.exists():
   r=json.loads(p.read_text());e.update(status=r['status'],valid_windows=r['valid_windows'],contact_windows=r['contact_windows'],report_result_sha256=sha256(p),completed_source_verified=r['status']=='accepted_success_source_verified')
   if r.get('source_audit'):e['source_sha256']=r['source_audit']['source_file_sha256']
  bank['entries'].append(e)
 for seed in range(100071,100087):bank['entries'].append({'scenario':'cwmj_wine_v2','seed':seed,'fold':'test','stage':'future_confirmation','role':'reserved_independent_confirmation','status':'reserved_not_executed','future_fit_requires_new_manifest':True})
 usage={}
 manifest_path=R/'data/reports/cwmj_m6/training_manifest.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text());allowed=set(manifest['train_seeds'])
  for repeat in range(3):
   root=R/f'data/reports/cwmj_m6_training_r{repeat}';progress=root/'progress.json';log=root/'training_log.json'
   if not progress.exists() or not log.exists():continue
   state=json.loads(progress.read_text());logs=json.loads(log.read_text());protocol=json.loads((root/'protocol.json').read_text())
   if protocol['training_manifest_sha256']!=sha256(manifest_path):raise ValueError('training provenance drift')
   for arm,update in state['updates'].items():
    if update['steps']!=720 or update['changed_tensors']<=0 or sha256(Path(update['artifact']))!=update['sha256']:raise ValueError('completed arm evidence mismatch')
    seen={r['seed'] for r in logs if r['arm']==arm}
    if not seen<=allowed:raise ValueError('heldout seed appears in optimizer log')
    for seed in seen:usage.setdefault(seed,[]).append({'repeat':repeat,'arm':arm,'artifact_sha256':update['sha256'],'training_log_sha256':sha256(log)})
 for e in bank['entries']:
  if e['scenario']=='cwmj_wine_v2':
   e['used_as_expert_supervision_in_m6']=e['seed'] in usage
   if e['seed'] in usage:
    if e['fold']!='train' or e['role']!='expert_collection_attempt':raise ValueError('training role leak')
    e['m6_completed_training_usage_evidence']=usage[e['seed']]
 for scenario in {e['scenario'] for e in bank['entries']}:
  folds={fold:{e['seed'] for e in bank['entries'] if e['scenario']==scenario and e['fold']==fold} for fold in ['train','validation','test']}
  if any(folds[a]&folds[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('cross-fold seed leakage')
 p=R/'datasets/tactile_seed_bank_m1_m6_v2.json';p.write_text(json.dumps(bank,indent=2));print('SEED_BANK_ENTRIES',len(bank['entries']))
if __name__=='__main__':main()

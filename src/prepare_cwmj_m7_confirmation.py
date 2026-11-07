"""Freeze M7 only after all three complete matched training runs; never choose best."""
from pathlib import Path
import json
from verify_m7_training import verify
from verify_cwmj_m7_confirmation import FILES
from m7_tactile_controls import CONDITIONS
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]
def prepare():
 training_manifest=verify(R)
 frozen_m6=json.loads((R/'data/reports/cwmj_m6/confirmation_manifest.json').read_text())
 weights={};evidence=['data/reports/cwmj_m7/training_manifest.json','datasets/tactile_seed_bank_m7_reservations_v1.json']
 for repeat in range(3):
  folder=R/f'data/reports/cwmj_m7_training_r{repeat}'
  result=json.loads((folder/'result.json').read_text());p=json.loads((folder/'protocol.json').read_text())
  old=R/f'data/reports/cwmj_m6_training_r{repeat}';op=json.loads((old/'protocol.json').read_text());o=json.loads((old/'result.json').read_text())
  if result['status']!='completed_real_adapter_training_offline_pilot' or result['repeat_index']!=repeat:raise ValueError('complete training required')
  if p['initialization']!=op['initialization'] or p['transformed_dataset_sha256']!=op['transformed_dataset_sha256']:raise ValueError('matched training source gate')
  if result['results']['frozen']!=o['results']['frozen']:raise ValueError('frozen model validation drift')
  if p['code_sha256']!=training_manifest['files']['src/run_cwmj_m7_training.py'] or p['protocol_sha256']!=training_manifest['files']['docs/CWMJ_M7_TRAINING_PROTOCOL_ZH.md'] or p['controls_sha256']!=training_manifest['files']['src/contact_state_control.py']:raise ValueError('training result code provenance mismatch')
  for arm in ('zero','true','mismatch'):
   prior=o['updates'][arm];pinned=frozen_m6['weights'][str(repeat)][arm]
   if prior['artifact']!=pinned['artifact'] or prior['sha256']!=pinned['sha256']:raise ValueError('M6 result differs from frozen final weight')
  update=result['updates']['contact_state']
  if update['steps']!=720 or update['changed_tensors']<=0 or sha256(Path(update['artifact']))!=update['sha256']:raise ValueError('invalid final M7 weight')
  weights[str(repeat)]={a:dict(artifact=(update if a=='contact_state' else o['updates'][a])['artifact'],sha256=(update if a=='contact_state' else o['updates'][a])['sha256']) for a in CONDITIONS}
  evidence += [str((folder/n).relative_to(R)) for n in ('result.json','protocol.json')]
 for seed in range(100087,100103):
  for parent in ('data/eval','data/paired'):
   if list((R/parent).glob(f'*{seed}*')):raise ValueError('confirmation seed already consumed')
 m=dict(seeds=list(range(100087,100103)),repetitions=[0,1,2],conditions=list(CONDITIONS),weights=weights,code_sha256={n:sha256(R/n) for n in FILES},evidence_sha256={n:sha256(R/n) for n in evidence})
 path=R/'data/reports/cwmj_m7/confirmation_manifest.json'
 with path.open('x') as f:json.dump(m,f,indent=2)
 print(path)
if __name__=='__main__':prepare()

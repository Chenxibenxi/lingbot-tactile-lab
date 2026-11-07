"""Require all M7 repetitions, fixed final weights, and untouched source pins."""
import json
from pathlib import Path
from validate_expert_data import sha256
from verify_m7_training import verify
from m7_tactile_controls import CONDITIONS
FILES=['src/run_cwmj_m7_confirmation.py','src/m7_tactile_controls.py','src/audit_cwmj_m7_pair.py','src/m7_paired_seed_statistics.py','src/report_cwmj_m7_confirmation.py','src/report_cwmj_m7_group.py','src/verify_cwmj_m7_confirmation.py','work/run_cwmj_m7_confirmation.sh','work/run_cwmj_m7_confirmation_batch.py','src/report_cwmj_m7_confirmation_batch.py','src/m7_completion_inventory.py','src/report_m7_infrastructure_failure.py','src/prepare_cwmj_m7_confirmation.py','src/report_cwmj_m7_progress.py']
EVIDENCE=['data/reports/cwmj_m7/training_manifest.json','datasets/tactile_seed_bank_m7_reservations_v1.json']+[f'data/reports/cwmj_m7_training_r{r}/{n}' for r in range(3) for n in ('result.json','protocol.json')]
def verify_confirmation(repo,seed,repeat,condition):
 if type(seed) is not int or seed not in range(100087,100103) or type(repeat) is not int or repeat not in (0,1,2) or condition not in CONDITIONS:raise ValueError('fixed M7 matrix required')
 verify(repo)
 m=json.loads((repo/'data/reports/cwmj_m7/confirmation_manifest.json').read_text())
 if m['seeds']!=list(range(100087,100103)) or m['conditions']!=list(CONDITIONS) or m['repetitions']!=[0,1,2]:raise ValueError('M7 matrix drift')
 if set(m['code_sha256'])!=set(FILES):raise ValueError('incomplete M7 code pins')
 for n,d in m['code_sha256'].items():
  if sha256(repo/n)!=d:raise ValueError('M7 confirmation code drift: '+n)
 if set(m['evidence_sha256'])!=set(EVIDENCE):raise ValueError('incomplete M7 evidence pins')
 for n,d in m['evidence_sha256'].items():
  if sha256(repo/n)!=d:raise ValueError('M7 confirmation evidence drift: '+n)
 w=m['weights'][str(repeat)][condition]
 if sha256(Path(w['artifact']))!=w['sha256']:raise ValueError('final weight drift')
 if list((repo/'data/paired').glob(f'*{seed}*')):raise ValueError('test seed used for collection')
 if any(not p.name.startswith('cwmj_m7_confirm_') for p in (repo/'data/eval').glob(f'*{seed}*')):raise ValueError('test seed earlier use')
 return m

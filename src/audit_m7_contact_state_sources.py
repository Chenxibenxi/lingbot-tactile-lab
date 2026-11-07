"""Audit every M6 train/validation input for the new contact-only contract."""
from pathlib import Path
import json
import torch
from cwmj_m4_samples import M4TerminalEpisode
from cwmj_m6_samples import M6TerminalEpisode
from contact_state_control import contact_state_payload
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]
def audit():
 mp=R/'data/reports/cwmj_m6/training_manifest.json';manifest=json.loads(mp.read_text());rows=[]
 for seed in manifest['train_seeds']+manifest['validation_seeds']:
  ep=M4TerminalEpisode(seed) if seed in (100045,100046,100049) else M6TerminalEpisode(seed)
  expected=next(s for s in manifest['sources'] if s['seed']==seed)
  if ep.hashes!=expected['source_sha256']:raise ValueError('source drift')
  contacts=0
  for i in range(len(ep)):
   s=ep.sample(i,False);x=s['input']
   p=dict(tactile_history=torch.as_tensor(x['tactile_history']).unsqueeze(0),tactile_mask=torch.as_tensor(x['tactile_mask']).unsqueeze(0),tactile_age_ms=torch.tensor([x['tactile_age_ms']]),tactile_valid=torch.tensor([True]))
   q=contact_state_payload(p);bit=bool(q['tactile_history'][...,2].any())
   for k in ('tactile_mask','tactile_age_ms','tactile_valid'):
    assert torch.equal(p[k],q[k])
   assert not q['tactile_history'][...,:2].any()
   if bit!=bool(s['metadata']['contact']):raise ValueError('M6 contact definition mismatch')
   contacts+=bit
  rows.append(dict(seed=seed,fold=ep.fold,windows=len(ep),contact_windows=contacts,source_sha256=ep.hashes))
  print('AUDITED',seed,len(ep),contacts,flush=True)
 out=R/'data/reports/m7_contact_state_source_audit.json';out.write_text(json.dumps(dict(status='passed',scope='CPU source and transformation audit, not training or closed-loop benefit',parent_manifest_sha256=sha256(mp),control_sha256=sha256(R/'src/contact_state_control.py'),audit_sha256=sha256(Path(__file__)),sources=rows),indent=2))
 print(out)
if __name__=='__main__':audit()

"""M7 training source pinning; no GPU allocation."""
import json
from validate_expert_data import sha256
from verify_cwmj_m6_confirmation import verify_confirmation

def verify(repo):
 m=json.loads((repo/'data/reports/cwmj_m7/training_manifest.json').read_text())
 for name,digest in m['files'].items():
  if sha256(repo/name)!=digest:raise ValueError('M7 source drift: '+name)
 for r in range(3):
  for c in ('zero','true','mismatch'):verify_confirmation(repo,100071,r,c)
 if m['confirmation_seeds']!=list(range(100087,100103)):raise ValueError('new confirmation seed drift')
 return m

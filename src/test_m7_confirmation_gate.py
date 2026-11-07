import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import verify_cwmj_m7_confirmation as v
class GateTest(unittest.TestCase):
 def fixture(self,root):
  r=Path(root)
  for n in v.FILES+v.EVIDENCE:
   p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
  w=r/'weight';w.write_text('weight')
  m=dict(seeds=list(range(100087,100103)),repetitions=[0,1,2],conditions=list(v.CONDITIONS),code_sha256={n:v.sha256(r/n) for n in v.FILES},evidence_sha256={n:v.sha256(r/n) for n in v.EVIDENCE},weights={str(i):{a:dict(artifact=str(w),sha256=v.sha256(w)) for a in v.CONDITIONS} for i in range(3)})
  p=r/'data/reports/cwmj_m7/confirmation_manifest.json';p.write_text(json.dumps(m));return r,m,p,w
 def test_reject_code_and_weight_drift(self):
  with tempfile.TemporaryDirectory() as d,patch.object(v,'verify'):
   r,m,p,w=self.fixture(d);v.verify_confirmation(r,100087,0,'true')
   w.write_text('changed')
   with self.assertRaises(ValueError):v.verify_confirmation(r,100087,0,'true')
   w.write_text('weight');(r/v.FILES[0]).write_text('changed')
   with self.assertRaises(ValueError):v.verify_confirmation(r,100087,0,'true')
 def test_missing_evidence_pin(self):
  with tempfile.TemporaryDirectory() as d,patch.object(v,'verify'):
   r,m,p,w=self.fixture(d);m['evidence_sha256'].pop(v.EVIDENCE[-1]);p.write_text(json.dumps(m))
   with self.assertRaises(ValueError):v.verify_confirmation(r,100087,0,'true')
 def test_old_seed_or_prior_use(self):
  with tempfile.TemporaryDirectory() as d,patch.object(v,'verify'):
   r,m,p,w=self.fixture(d)
   with self.assertRaises(ValueError):v.verify_confirmation(r,100071,0,'true')
   (r/'data/eval/earlier_100087').mkdir(parents=True)
   with self.assertRaises(ValueError):v.verify_confirmation(r,100087,0,'true')
if __name__=='__main__':unittest.main()

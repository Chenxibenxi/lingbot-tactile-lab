import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import prepare_cwmj_m7_confirmation as p
class PrepareTest(unittest.TestCase):
 def fixture(self,root):
  r=Path(root);m={'files':{'src/run_cwmj_m7_training.py':'code','docs/CWMJ_M7_TRAINING_PROTOCOL_ZH.md':'protocol','src/contact_state_control.py':'control'}}
  for n in p.FILES+['data/reports/cwmj_m7/training_manifest.json','datasets/tactile_seed_bank_m7_reservations_v1.json']:
   f=r/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture')
  weights={}
  for i in range(3):
   new=r/f'data/reports/cwmj_m7_training_r{i}';old=r/f'data/reports/cwmj_m6_training_r{i}';new.mkdir();old.mkdir();w=new/'weight';w.write_text('weight')
   u=dict(artifact=str(w),sha256=p.sha256(w),steps=720,changed_tensors=5)
   o=dict(updates={a:u for a in ('zero','true','mismatch')},results={'frozen':[1]});weights[str(i)]={a:dict(artifact=str(w),sha256=p.sha256(w)) for a in o['updates']}
   n=dict(status='completed_real_adapter_training_offline_pilot',repeat_index=i,updates={'contact_state':u},results={'frozen':[1]})
   q=dict(initialization=str(i),transformed_dataset_sha256='same',code_sha256='code',protocol_sha256='protocol',controls_sha256='control')
   for folder,result in ((new,n),(old,o)):
    (folder/'result.json').write_text(json.dumps(result));(folder/'protocol.json').write_text(json.dumps(q))
  f=r/'data/reports/cwmj_m6/confirmation_manifest.json';f.parent.mkdir();f.write_text(json.dumps(dict(weights=weights)))
  return r,m
 def test_success_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   r,m=self.fixture(d)
   with patch.object(p,'R',r),patch.object(p,'verify',return_value=m):
    p.prepare()
    with self.assertRaises(FileExistsError):p.prepare()
 def test_old_mapping_or_training_code_drift(self):
  for mode in ('old','code'):
   with tempfile.TemporaryDirectory() as d:
    r,m=self.fixture(d)
    f=r/('data/reports/cwmj_m6_training_r1/result.json' if mode=='old' else 'data/reports/cwmj_m7_training_r1/protocol.json');q=json.loads(f.read_text())
    if mode=='old':q['updates']['true']['sha256']='changed'
    else:q['code_sha256']='changed'
    f.write_text(json.dumps(q))
    with patch.object(p,'R',r),patch.object(p,'verify',return_value=m):
     with self.assertRaises(ValueError):p.prepare()
    self.assertFalse((r/'data/reports/cwmj_m7/confirmation_manifest.json').exists())
if __name__=='__main__':unittest.main()

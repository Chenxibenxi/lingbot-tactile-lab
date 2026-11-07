import unittest,tempfile,json
from pathlib import Path
import h5py,numpy as np
from audit_cwmj_m7_pair import audit_pair,CONDITIONS
class PairTest(unittest.TestCase):
 def fixture(self,root):
  trials={}
  for a in CONDITIONS:
   p=Path(root)/a;p.mkdir();trials[a]=p
   (p/'result.json').write_text(json.dumps(dict(arm=a,status='completed',success=False,termination='official_action_limit',seed=100087,repeat_index=0,adapter_sha256=a)))
   (p/'scenario.json').write_text('{}')
   with h5py.File(p/'rollout.hdf5','w') as h:
    for k in ('images/head','images/left','images/right'):h[k]=np.zeros((1,2,2,3),dtype='uint8')
    h['state_14']=np.zeros((1,14));h['applied/command_14']=np.zeros((20,14));h['tactile/window_valid']=np.zeros((1,1),dtype=bool)
  return trials
 def test_separate_weights_and_postinitial_divergence_allowed(self):
  with tempfile.TemporaryDirectory() as d:
   t=self.fixture(d)
   with h5py.File(t['contact_state']/'rollout.hdf5','a') as h:h['applied/command_14'][10:]=1
   self.assertTrue(audit_pair(t,100087,0)['eligible_complete_pair'])
 def test_initial_action_drift_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   t=self.fixture(d)
   with h5py.File(t['contact_state']/'rollout.hdf5','a') as h:h['applied/command_14'][0]=1
   self.assertFalse(audit_pair(t,100087,0)['eligible_complete_pair'])
 def test_condition_identity_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   t=self.fixture(d);p=t['true']/'result.json';r=json.loads(p.read_text());r['arm']='zero';p.write_text(json.dumps(r))
   with self.assertRaises(ValueError):audit_pair(t,100087,0)
if __name__=='__main__':unittest.main()

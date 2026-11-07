import unittest,tempfile,json
from pathlib import Path
import numpy as np,h5py
from audit_cwmj_m6_pair import audit_pair,CONDITIONS
class PairAudit(unittest.TestCase):
 def test_equal_then_precontact_corruption_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   paths={a:Path(d)/a for a in CONDITIONS}
   for a,p in paths.items():
    p.mkdir();(p/'scenario.json').write_text('{"asset":"fixed"}');(p/'result.json').write_text(json.dumps({'status':'completed','termination':'official_action_limit','success':False,'adapter_sha256':'same','requests':[{'request_index':i,'past_contact':i==2} for i in range(3)]}))
    with h5py.File(p/'rollout.hdf5','w') as h:
     h['state_14']=np.zeros((3,14));h['applied/command_14']=np.zeros((30,14));h['tactile/window_valid']=np.array([[False],[True],[True]])
     for k in ['head','left','right']:h['images/'+k]=np.zeros((3,2,2,3))
   self.assertTrue(audit_pair(paths)['eligible_complete_pair'])
   with h5py.File(paths['true_input_zero']/'rollout.hdf5','r+') as h:h['applied/command_14'][15,0]=1
   self.assertFalse(audit_pair(paths)['technical_comparable'])
 def test_missing_rollout_not_passing_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   paths={a:Path(d)/a for a in CONDITIONS}
   for p in paths.values():p.mkdir();(p/'result.json').write_text('{"status":"failed"}')
   self.assertFalse(audit_pair(paths)['technical_comparable'])

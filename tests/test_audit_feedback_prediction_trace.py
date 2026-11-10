import json,tempfile,unittest
from pathlib import Path
import h5py
import numpy as np
from feedback_prediction_trace import PredictionTrace
from audit_feedback_prediction_trace import audit
class AuditTest(unittest.TestCase):
 def test_accept_and_detect_tampered_command(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);state={s:dict(arm_actual_qpos=[0]*6,arm_drive_targets=[0]*6,entity_actual_qpos=[0]*6,active_joint_dofs=[1]*6,active_joint_names=list('abcdef')) for s in ('left','right')}
   w=PredictionTrace(p/'prediction_trace.jsonl');w.request(index=0,physics_step=0,commands=np.zeros((50,14)),physical_state=state)
   w.action(command=np.zeros(14),physics_before=0,physics_after=2,before=state,after=state);w.close(complete=True)
   (p/'result.json').write_text(json.dumps(dict(status='completed',resource_censored=False,actions=1,requests=[{}])))
   with h5py.File(p/'rollout.hdf5','x') as f:
    f['applied/command_14']=np.zeros((1,14))
    for n,v in [('physics_step_before',[0]),('physics_step_after',[2]),('request_index',[0]),('index_in_request',[0])]:f['applied/'+n]=v
    f['request/physics_step']=[0]
   self.assertEqual(audit(p)['actions'],1)
   with h5py.File(p/'rollout.hdf5','r+') as f:f['applied/command_14'][0,0]=1
   with self.assertRaises(AssertionError):audit(p)
if __name__=='__main__':unittest.main()

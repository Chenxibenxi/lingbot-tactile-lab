import json,tempfile,unittest
from pathlib import Path
import numpy as np
import h5py
from feedback_block_trace import BlockTrace
from audit_feedback_block_trace import audit

class TestBlockTrace(unittest.TestCase):
 def test_roundtrip_and_tamper(self):
  for mode in ('fresh_first','cached_block'):
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'trace';h=Path(d)/'rollout';w=BlockTrace(p,mode);commands=[]
    for i in range(12):
     b=np.repeat((100*i+np.arange(50,dtype=float))[:,None],14,axis=1)
     c,_=w.request(index=i,physics_step=2*i,commands=b,physical_state={})
     w.action(command=c,physics_before=2*i,physics_after=2*i+2,before={},after={});commands.append(c)
    w.close(complete=True)
    with h5py.File(h,'w') as f:
     f.attrs['complete']=True
     for key,value in {'request/physics_step':np.arange(12)*2,'applied/command_14':np.array(commands,dtype=np.float32),'applied/request_index':np.arange(12),'applied/index_in_request':np.zeros(12,dtype=int),'applied/physics_step_before':np.arange(12)*2,'applied/physics_step_after':np.arange(12)*2+2}.items():f[key]=value
    self.assertEqual(audit(p,h)['actions'],12)
    original=p.read_text()
    for field in ('source','command','clock'):
     rows=[json.loads(s) for s in original.splitlines()]
     if field=='source':rows[4]['source']['source_request_index']=11
     elif field=='command':rows[4]['command'][0]+=1
     else:rows[4]['physics_after']+=1
     p.write_text('\n'.join(map(json.dumps,rows)))
     with self.assertRaises(ValueError):audit(p,h)
    p.write_text(original)
    with h5py.File(h,'r+') as f:f['applied/command_14'][1,0]+=1
    with self.assertRaises(ValueError):audit(p,h)
 def test_pending_and_clock(self):
  with tempfile.TemporaryDirectory() as d:
   w=BlockTrace(Path(d)/'trace','cached_block')
   c,_=w.request(index=0,physics_step=0,commands=np.zeros((50,14)),physical_state={})
   with self.assertRaises(ValueError):w.close(complete=True)
   with self.assertRaises(ValueError):w.action(command=c,physics_before=1,physics_after=2,before={},after={})
   w.close()

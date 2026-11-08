"""Exercise journal + resource race + guarded entry as a single dispatch path."""
import json,tempfile,unittest
from pathlib import Path
from durable_resource_queue import run
from guarded_gpu_entry import enter
class Integration(unittest.TestCase):
 def test_waiter_arriving_after_preflight_pauses_with_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);script=root/'model.py';script.write_text('raise AssertionError("must not load")')
   free=dict(probe_ok=True,compute_pids=[],lease_status={'state':'free','waiting':[]})
   after=dict(probe_ok=True,compute_pids=[],ancestors=[{'ppid':12}],lease_status=dict(state='held',holder='ours',pid_alive=True,lease={'pid':12},remaining_s=2100,waiting=[{'owner':'secretary'}]))
   calls=[]
   def launch(item):
    calls.append(item)
    rc=enter(script,[],root/f'{item}_admission.json',owner='ours',read_snapshot=lambda:after)
    return dict(status='failed',returncode=rc,resource_censored=None)
   result=run(root/'queue',['a','b','c'],owner='ours',launch=launch,resource_snapshot=lambda:free)
   self.assertEqual(calls,['a']);self.assertEqual(result['status'],'paused')
   self.assertEqual(json.loads((root/'a_admission.json').read_text())['reasons'],['other_gpu_waiter'])
   self.assertFalse((root/'queue/started/b.json').exists())
   with self.assertRaises(ValueError):run(root/'queue',['a','b','c'],owner='ours',launch=launch,resource_snapshot=lambda:free)
   self.assertEqual(calls,['a'])

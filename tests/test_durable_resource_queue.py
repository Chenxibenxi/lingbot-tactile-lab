import tempfile,unittest,json,fcntl
from pathlib import Path
from durable_resource_queue import run
class Queue(unittest.TestCase):
 def test_wait_then_resume_without_consuming_or_repeating(self):
  with tempfile.TemporaryDirectory() as d:
   calls=[];free=dict(probe_ok=True,compute_pids=[],lease_status={'state':'free','waiting':[]})
   busy=dict(free,compute_pids=[7])
   def launch(x):calls.append(x);return dict(returncode=0,status='completed',resource_censored=False,success=False)
   kw=dict(launch=launch,owner='ours')
   self.assertEqual(run(d,['a','b'],resource_snapshot=lambda:busy,**kw)['status'],'paused')
   self.assertEqual(calls,[])
   run(d,['a','b'],resource_snapshot=lambda:free,**kw);run(d,['a','b'],resource_snapshot=lambda:free,**kw)
   self.assertEqual(calls,['a','b'])
 def test_failure_persists_and_cannot_be_retried(self):
  with tempfile.TemporaryDirectory() as d:
   free=lambda:dict(probe_ok=True,compute_pids=[],lease_status={'state':'free','waiting':[]})
   calls=[]
   def launch(x):calls.append(x);raise RuntimeError('lease refusal')
   kw=dict(launch=launch,owner='ours',resource_snapshot=free)
   self.assertEqual(run(d,['a','b'],**kw)['status'],'paused')
   with self.assertRaises(ValueError):run(d,['a','b'],**kw)
   self.assertEqual(calls,['a'])
 def test_concurrent_dispatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with (Path(d)/'queue.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with self.assertRaises(BlockingIOError):run(d,['a'],launch=lambda x:None,owner='ours')

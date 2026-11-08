import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_feedback_confirmation_batch_v3 as batch
class Queue(unittest.TestCase):
 def test_real_batch_launcher_failure_stops_and_cannot_retry(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'data/reports/feedback_confirmation_v3').mkdir(parents=True);(r/'data/eval/feedback_confirmation_v3').mkdir(parents=True)
   free=dict(probe_ok=True,compute_pids=[],lease_status={'state':'free','waiting':[]})
   with patch.object(batch,'R',r),patch.object(batch,'verify'),patch.object(batch,'snapshot',return_value=free),patch.object(batch,'report'),patch.object(batch.subprocess,'run',return_value=SimpleNamespace(returncode=75)) as launch:
    result=batch.run();self.assertEqual(result['status'],'paused');self.assertEqual(launch.call_count,1)
    with self.assertRaises(ValueError):batch.run()
    self.assertEqual(launch.call_count,1)
 def test_real_batch_busy_gpu_creates_no_trial(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'data/reports/feedback_confirmation_v3').mkdir(parents=True)
   busy=dict(probe_ok=True,compute_pids=[123],lease_status={'state':'held','waiting':[]})
   with patch.object(batch,'R',r),patch.object(batch,'verify'),patch.object(batch,'snapshot',return_value=busy),patch.object(batch.subprocess,'run') as launch:
    self.assertEqual(batch.run()['status'],'paused');launch.assert_not_called()
    self.assertFalse((r/'data/eval').exists())

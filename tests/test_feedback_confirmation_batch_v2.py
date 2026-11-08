import tempfile
from pathlib import Path
import unittest
from run_feedback_confirmation_batch_v2 import matrix,write

class Batch(unittest.TestCase):
    def test_fixed_order_and_complete_unique_matrix(self):
        rows=matrix();self.assertEqual(len(rows),384);self.assertEqual(len(set(rows)),384)
        self.assertEqual(rows[0],(100129,0,10,'zero'))
        self.assertEqual(rows[-1],(100144,2,1,'contact_state'))
    def test_terminal_record_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.json';write(p,{'status':'failed'})
            with self.assertRaises(FileExistsError):write(p,{'status':'completed'})
            self.assertIn('failed',p.read_text())

class BatchFailurePaths(unittest.TestCase):
    def test_launcher_failures_are_unknown_without_retries(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        import run_feedback_confirmation_batch_v2 as b
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);logs=root/'data/reports/feedback_confirmation_v2';logs.mkdir(parents=True)
            (logs/'manifest.json').write_text('{}')
            (root/'data/eval/feedback_confirmation_v2').mkdir(parents=True)
            final=root/'data/reports/feedback_confirmation_v2_final';final.mkdir();(final/'report.md').write_text('synthetic report')
            launches=[]
            def call(args,**kwargs):
                if args[0]=='nvidia-smi':return SimpleNamespace(stdout='',returncode=0)
                launches.append(args);return SimpleNamespace(returncode=124)
            with patch.object(b,'R',root),patch.object(b,'verify'),patch.object(b,'episode_report'),patch.object(b,'final_report',return_value={'technical_gate':False}),patch.object(b.subprocess,'run',side_effect=call),patch.object(b.shutil,'disk_usage',return_value=SimpleNamespace(free=30*1024**3)):
                b.run();self.assertEqual(len(launches),384)
                queue=json.loads((logs/'queue_result.json').read_text())
                self.assertEqual(len(queue['attempts']),384)
                self.assertTrue(all(x['success'] is None and x['status']=='failed' for x in queue['attempts']))
                with self.assertRaises(FileExistsError):b.run()
                self.assertEqual(len(launches),384)
    def test_busy_gpu_prevents_launch(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        import run_feedback_confirmation_batch_v2 as b
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);logs=root/'data/reports/feedback_confirmation_v2';logs.mkdir(parents=True)
            (logs/'manifest.json').write_text('{}')
            with patch.object(b,'R',root),patch.object(b,'verify'),patch.object(b.subprocess,'run',return_value=SimpleNamespace(stdout='123')) as call:
                b.run();self.assertEqual(call.call_count,1)
            self.assertEqual(json.loads((logs/'queue_result.json').read_text())['status'],'paused')

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_controlled_feedback_batch as batch


class BatchTest(unittest.TestCase):
    def test_launcher_failure_keeps_fixed_matrix_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);logs=root/'data/reports/controlled_feedback_pilot_v1';logs.mkdir(parents=True)
            (root/'data/eval/controlled_feedback_pilot_v1').mkdir(parents=True)
            launches=[]
            def run(args,**kwargs):
                if args[0]=='nvidia-smi':return SimpleNamespace(stdout='',returncode=0)
                launches.append(args);return SimpleNamespace(returncode=1)
            with patch.object(batch,'R',root),patch.object(batch,'verify'),patch.object(batch,'report'), \
                 patch.object(batch.subprocess,'run',side_effect=run), \
                 patch.object(batch.shutil,'disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
                batch.run()
                self.assertEqual(len(launches),16)
                data=json.loads((logs/'queue_result.json').read_text())
                self.assertEqual(len(data['attempts']),16)
                self.assertTrue(all(a['success'] is None for a in data['attempts']))
                with self.assertRaises(FileExistsError):batch.run()
                self.assertEqual(len(launches),16)

    def test_busy_gpu_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);logs=root/'data/reports/controlled_feedback_pilot_v1';logs.mkdir(parents=True)
            with patch.object(batch,'R',root),patch.object(batch,'verify'), \
                 patch.object(batch.subprocess,'run',return_value=SimpleNamespace(stdout='1234')) as run:
                batch.run();self.assertEqual(run.call_count,1)
            self.assertEqual(json.loads((logs/'queue_result.json').read_text())['status'],'paused')

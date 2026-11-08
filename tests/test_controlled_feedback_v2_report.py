import json
from pathlib import Path
import tempfile
import unittest
from report_controlled_feedback_pilot_v2 import report


class ReportTest(unittest.TestCase):
    def test_failed_before_capture_keeps_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);trial=root/'trial';trial.mkdir()
            (trial/'result.json').write_text(json.dumps(dict(status='failed',error='initialization failed',requests=[])))
            out=root/'report';report(trial,out)
            text=(out/'report.md').read_text()
            self.assertIn('官方成功：未知',text)
            self.assertIn('不补造画面',text)
            self.assertFalse((out/'camera_frames.png').exists())
            self.assertFalse(json.loads((out/'summary.json').read_text())['confirmation_eligible'])

    def test_running_result_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'result.json').write_text(json.dumps(dict(status='running')))
            with self.assertRaises(ValueError):report(root,root/'report')
            self.assertFalse((root/'report').exists())

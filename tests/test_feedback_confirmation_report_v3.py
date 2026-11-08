import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
from report_feedback_confirmation_final_v3 import report,SEEDS,ARMS

class ConfirmationReport(unittest.TestCase):
    def test_unknowns_retained_and_missing_episode_reports_disclosed(self):
        rows=[dict(seed=s,repeat=r,cadence=c,arm=a,success=None,technical_pass=False,error='synthetic unknown') for s in SEEDS for r in (0,1,2) for c in (10,1) for a in ARMS]
        fixture=dict(technical_gate=False,unknown_trials=384,episodes=rows,statistics=None)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data/reports').mkdir(parents=True)
            with patch('report_feedback_confirmation_final_v3.audit',return_value=fixture):report(root)
            out=root/'data/reports/feedback_confirmation_v3_final'
            text=(out/'report.md').read_text()
            self.assertEqual(text.count('| None | False |'),384)
            self.assertIn('逐场图文缺失：**384**',text)
            self.assertTrue((out/'matrix.png').stat().st_size>1000)

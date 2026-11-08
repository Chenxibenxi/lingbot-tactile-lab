from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from report_feedback_pilot_final import report


class FinalReportTest(unittest.TestCase):
    def test_unknown_matrix_remains_unknown_and_not_approved(self):
        fixture=dict(technical_gate=False,initial_first_command_pair_checks={'100127':False,'100128':False},
            episodes=[dict(seed=s,cadence=c,arm=a,success=None,status='failed',technical_pass=False,reason='synthetic missing data')
                for s in (100127,100128) for c in (10,1) for a in ('zero','true','mismatch','contact_state')])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data/reports').mkdir(parents=True)
            with patch('report_feedback_pilot_final.audit',return_value=fixture):report(root)
            out=root/'data/reports/controlled_feedback_pilot_v1_final'
            text=(out/'report.md').read_text()
            self.assertEqual(text.count('| None | False |'),16)
            self.assertIn('技术通过门：**False**',text)
            self.assertTrue((out/'matrix.png').is_file())

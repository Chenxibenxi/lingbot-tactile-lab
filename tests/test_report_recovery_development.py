import json
from pathlib import Path
import tempfile
import unittest
from report_recovery_development import report


class RecoveryReport(unittest.TestCase):
    def test_interruption_report_preserves_unknown_and_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);trial=root/'trial';trial.mkdir()
            (trial/'failure.json').write_text(json.dumps(dict(status='needs_inspection',seed=100119,
                condition='left_open_once_recovery',error_type='RuntimeError',error='synthetic interrupted controller')))
            event=dict(event='before_action',action_index=0,before_physics_step=0,camera_frame_index=0,
                       command_state_14=[0.]*14,source_role='expert',training_approval=False)
            (trial/'action_journal.jsonl').write_text(json.dumps(event)+'\n')
            out=root/'report';report(trial,out,publish=False)
            result=json.loads((out/'summary.json').read_text())
            self.assertEqual(result['unresolved_action_attempts'],[0])
            self.assertEqual(result['completed_action_markers'],0)
            self.assertNotIn('official_success',result['result'])
            self.assertGreater((out/'actions_contact.png').stat().st_size,1000)
            self.assertIn('未知，未取得可评分终态',(out/'report.md').read_text())

    def test_live_attempt_has_no_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):report(root,root/'report',publish=False)
            self.assertFalse((root/'report').exists())

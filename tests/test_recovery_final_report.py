import json
from pathlib import Path
import tempfile
import unittest
from report_recovery_v2_final import report


class FinalReportTest(unittest.TestCase):
    def test_live_or_incomplete_queue_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'data/reports/recovery_development_v2';p.mkdir(parents=True)
            for data in ({'status':'running'},{'status':'completed','attempts':[]}):
                (p/'queue_result.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError):report(root)
            self.assertFalse((root/'data/reports/recovery_development_v2_final').exists())

    def test_missing_data_does_not_pass_or_shrink_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'data/reports/recovery_development_v2';p.mkdir(parents=True)
            attempts=[dict(seed=s,kind=k,status='prerequisite_failed') for s in range(100123,100127)
                for k in ('prefix_generation','expert_control','left_open_once_recovery')]
            (p/'queue_result.json').write_text(json.dumps(dict(status='completed',attempts=attempts)))
            result=report(root)
            self.assertFalse(result['development_gate'])
            self.assertEqual(result['fixed_seed_denominator'],4)
            self.assertEqual(result['technical_complete_pairs'],0)
            self.assertTrue(all(a['outcomes']['expert_control'] is None for a in result['audits']))

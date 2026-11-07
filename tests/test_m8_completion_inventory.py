import json
from pathlib import Path
import tempfile
import unittest
from m8_completion_inventory import inventory,planned_attempts


class InventoryTest(unittest.TestCase):
    def test_all_planned_and_wrong_duration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(len(inventory(root)['pending']),384)
            for name,(seed,r,s,a) in planned_attempts().items():
                p=root/f'data/eval/cwmj_m8_confirm_{name}';p.mkdir(parents=True)
                (p/'result.json').write_text(json.dumps(dict(seed=seed,repeat_index=r,training_steps=s,
                    arm=a,status='completed',success=False,termination='official_action_limit')))
            self.assertTrue(inventory(root)['ready_for_final_analysis'])
            p=root/'data/eval/cwmj_m8_confirm_r0_s720_100103_zero/result.json'
            value=json.loads(p.read_text());value['training_steps']=2880;p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):inventory(root)

    def test_missing_is_not_failure(self):
        with tempfile.TemporaryDirectory() as d:
            result=inventory(Path(d))
            self.assertFalse(result['ready_for_final_analysis'])
            self.assertEqual(result['terminal'],{})

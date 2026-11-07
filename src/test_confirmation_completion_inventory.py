import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from confirmation_completion_inventory import inventory, CONDITIONS


class CompletionInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'data/reports').mkdir(parents=True)
        for arm in CONDITIONS:
            p = self.root / f'data/eval/cwmj_m6_confirm_r0_100080_{arm}'
            p.mkdir(parents=True)
            if arm == 'true_input_zero':
                raw = p / 'partial.bin'
                raw.write_bytes(b'preserved partial data')
                record = dict(interrupted_attempt=p.name.removeprefix('cwmj_m6_confirm_'),
                              original_files_sha256={str(raw.relative_to(self.root)): hashlib.sha256(raw.read_bytes()).hexdigest()})
                (self.root / 'data/reports/cwmj_m6_confirmation_interruption_test.json').write_text(json.dumps(record))
            else:
                (p / 'result.json').write_text(json.dumps(dict(seed=100080, arm=arm, repeat_index=0,
                    status='completed', success=False, termination='official_action_limit')))

    def check(self):
        return inventory(self.root, seeds=[100080], repeats=[0])

    def test_interruption_is_not_failure_and_excludes_whole_seed(self):
        result = self.check()
        self.assertTrue(result['ready_for_final_analysis'])
        self.assertEqual(len(result['terminal']), 3)
        self.assertEqual(result['interrupted_seeds'], [100080])

    def test_unattempted_condition_blocks_final_analysis(self):
        next((self.root / 'data/eval').glob('*_zero/result.json')).unlink()
        self.assertFalse(self.check()['ready_for_final_analysis'])

    def test_modified_partial_evidence_is_rejected(self):
        next((self.root / 'data/eval').glob('*/partial.bin')).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.check()

    def test_interrupted_attempt_cannot_be_replaced_by_result(self):
        p = next((self.root / 'data/eval').glob('*/partial.bin')).parent
        (p / 'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'unexpectedly has result'):
            self.check()


if __name__ == '__main__':
    unittest.main()

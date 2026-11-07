"""Synthetic complete matrix exercises whole-seed exclusion without GPU use."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import report_cwmj_m7_confirmation_batch as report


class InterruptedReportTest(unittest.TestCase):
    def test_complete_matrix_preserves_unknown_and_excludes_whole_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/reports/cwmj_m7').mkdir(parents=True)
            (root / 'src').mkdir()
            (root / 'src/m7_completion_inventory.py').write_text('fixture')
            manifest = {'weights': {str(i): {a: {'sha256': 'fixture'} for a in ['zero', 'true', 'mismatch', 'contact_state']} for i in range(3)}}
            mp = root / 'data/reports/cwmj_m7/confirmation_manifest.json'
            mp.write_text(json.dumps(manifest))
            digest = hashlib.sha256(mp.read_bytes()).hexdigest()
            for seed in range(100087, 100103):
                for repeat in range(3):
                    for arm in report.CONDITIONS:
                        name = f'r{repeat}_{seed}_{arm}'
                        p = root / f'data/eval/cwmj_m7_confirm_{name}'
                        p.mkdir(parents=True)
                        if name == 'r0_100096_contact_state':
                            raw = p / 'partial'; raw.write_bytes(b'partial')
                            evidence = {'interrupted_attempt': name, 'original_files_sha256': {str(raw.relative_to(root)): hashlib.sha256(raw.read_bytes()).hexdigest()}}
                            (root / 'data/reports/cwmj_m7_confirmation_interruption_fixture.json').write_text(json.dumps(evidence))
                            continue
                        result = dict(seed=seed, repeat_index=repeat, arm=arm, status='completed', success=False, termination='official_action_limit', actions=400, adapter_sha256='fixture', confirmation_manifest_sha256=digest, requests=[])
                        (p / 'result.json').write_text(json.dumps(result))
                        (p / 'scenario.json').write_text('{}')
                        with h5py.File(p / 'rollout.hdf5', 'w') as h:
                            for key in ['images/head', 'images/left', 'images/right']:
                                h[key] = np.zeros((1, 2, 2, 3), dtype=np.uint8)
                            h['state_14'] = np.zeros((1, 14))
                            h['applied/command_14'] = np.zeros((10, 14))
                            h['tactile/window_valid'] = np.zeros((1, 1), dtype=bool)
            with patch.object(report, 'R', root), patch.object(report, 'DEST', root / 'published'), patch.object(report, 'verify_confirmation'):
                report.report()
            summary = json.loads((root / 'published/summary.json').read_text())
            self.assertEqual(summary['terminal_records'], 191)
            self.assertEqual([x['seed'] for x in summary['exclusions']], [100096])
            unknown = [x for x in summary['all_attempts'] if x['success'] is None]
            self.assertEqual(len(unknown), 1)
            self.assertFalse(unknown[0]['full'])
            self.assertIn('infrastructure_failure.md', (root / 'published/report.md').read_text())
            self.assertFalse((root / 'data/eval/cwmj_m7_confirm_r0_100096_contact_state/result.json').exists())


if __name__ == '__main__':
    unittest.main()

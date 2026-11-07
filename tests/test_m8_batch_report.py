"""Synthetic pipeline check; provenance verifier separately required on real runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import report_cwmj_m8_confirmation_batch as module
from m8_completion_inventory import planned_attempts
from validate_expert_data import sha256


def fixture(root):
    (root/'data/reports/cwmj_m8').mkdir(parents=True)
    manifest=root/'data/reports/cwmj_m8/confirmation_manifest.json';manifest.write_text('{"synthetic":true}')
    (root/'src').mkdir()
    (root/'src/m8_paired_seed_statistics.py').write_text('synthetic provenance placeholder')
    for name,(seed,r,s,a) in planned_attempts().items():
        p=root/f'data/eval/cwmj_m8_confirm_{name}';p.mkdir(parents=True)
        success=s==2880 and a=='true'
        (p/'result.json').write_text(json.dumps(dict(seed=seed,repeat_index=r,training_steps=s,arm=a,status='completed',
            success=success,termination='task_success' if success else 'official_action_limit',confirmation_manifest_sha256=sha256(manifest))))
        (p/'scenario.json').write_text('{"synthetic":true}')
        with h5py.File(p/'rollout.hdf5','w') as h:
            for cam in ('head','left','right'):h.create_dataset('images/'+cam,data=np.zeros((1,2,2,3),dtype=np.uint8))
            h.create_dataset('state_14',data=np.zeros((1,14)))
            h.create_dataset('applied/command_14',data=np.zeros((10,14)))
            h.create_dataset('tactile/window_valid',data=np.zeros((1,1),dtype=bool))


class BatchReport(unittest.TestCase):
    def test_full_pipeline_and_whole_seed_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            with h5py.File(root/'data/eval/cwmj_m8_confirm_r2_s2880_100118_true/rollout.hdf5','r+') as h:
                h['images/head'][0,0,0,0]=1
            with patch.object(module,'verify_confirmation') as verify:
                module.report(root,root/'published')
                self.assertEqual(verify.call_count,384)
            result=json.loads((root/'published/summary.json').read_text())
            self.assertEqual(len(result['eligible_seed_ids']),15)
            self.assertEqual(result['exclusions'][0]['seed'],100118)
            self.assertTrue(result['duration_explanation_supported'])
            self.assertEqual(result['terminal_count'],384)
            self.assertGreater((root/'published/matrix.png').stat().st_size,1000)
            self.assertTrue((root/'published/report.md').exists())

    def test_pending_prevents_final_report(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):module.report(Path(d),Path(d)/'published')
            self.assertFalse((Path(d)/'published').exists())

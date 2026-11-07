import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from audit_cwmj_m8_pair import audit_pair,audit_initial_across_nodes_and_repeats


class PairAudit(unittest.TestCase):
    def fixture(self,root):
        paths=[]
        for r in range(3):
            for steps in (720,2880):
                for arm in ('zero','true','mismatch','contact_state'):
                    p=root/f'{r}_{steps}_{arm}';p.mkdir();paths.append(p)
                    (p/'result.json').write_text(json.dumps(dict(seed=100103,repeat_index=r,training_steps=steps,arm=arm,
                        status='completed',success=False,termination='official_action_limit')))
                    (p/'scenario.json').write_text('{"asset":"synthetic"}')
                    with h5py.File(p/'rollout.hdf5','w') as h:
                        for camera in ('head','left','right'):h.create_dataset('images/'+camera,data=np.zeros((1,2,2,3),dtype=np.uint8))
                        h.create_dataset('state_14',data=np.zeros((1,14)))
                        h.create_dataset('applied/command_14',data=np.zeros((10,14)))
                        h.create_dataset('tactile/window_valid',data=np.zeros((1,1),dtype=bool))
        return paths

    def test_cross_node_camera_drift_detected(self):
        with tempfile.TemporaryDirectory() as d:
            paths=self.fixture(Path(d))
            self.assertTrue(audit_initial_across_nodes_and_repeats(paths))
            with h5py.File(paths[-1]/'rollout.hdf5','r+') as h:h['images/left'][0,0,0,0]=1
            self.assertFalse(audit_initial_across_nodes_and_repeats(paths))

    def test_missing_node_and_wrong_duration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            paths=self.fixture(Path(d))
            with self.assertRaises(ValueError):audit_initial_across_nodes_and_repeats(paths[:-1])
            trials=dict(zip(('zero','true','mismatch','contact_state'),paths[:4]))
            self.assertTrue(audit_pair(trials,100103,0,720)['eligible_complete_pair'])
            with self.assertRaises(ValueError):audit_pair(trials,100103,0,2880)

    def test_invalid_window_and_first_commands_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            paths=self.fixture(Path(d));trials=dict(zip(('zero','true','mismatch','contact_state'),paths[:4]))
            with h5py.File(paths[1]/'rollout.hdf5','r+') as h:
                h['tactile/window_valid'][0,0]=True
                h['applied/command_14'][0,0]=1
            result=audit_pair(trials,100103,0,720)
            self.assertFalse(result['eligible_complete_pair'])
            self.assertFalse(result['checks']['initial_invalid_all'])
            self.assertFalse(result['checks']['first10_commands_equal'])

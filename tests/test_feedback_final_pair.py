import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from audit_feedback_pilot_final import compare_initial


class PairTest(unittest.TestCase):
    def build(self,root):
        paths=[]
        for i in range(8):
            path=root/str(i);path.mkdir();paths.append(path)
            (path/'scenario.json').write_text(json.dumps({'asset':'same'}))
            with h5py.File(path/'rollout.hdf5','w') as h:
                for name in ('images/head','images/left','images/right'):
                    h[name]=np.zeros((1,2,2,3),dtype=np.uint8)
                h['state_14']=np.zeros((1,14))
                commands=np.zeros((2,14));commands[1]=i
                h['applied/command_14']=commands
                h['tactile/window_valid']=np.zeros((1,1),dtype=bool)
        return paths

    def test_later_commands_can_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(compare_initial(self.build(Path(directory))))

    def test_first_command_difference_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=self.build(Path(directory))
            with h5py.File(paths[-1]/'rollout.hdf5','a') as h:h['applied/command_14'][0,0]=.01
            self.assertFalse(compare_initial(paths))

    def test_missing_condition_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):compare_initial(self.build(Path(directory))[:-1])

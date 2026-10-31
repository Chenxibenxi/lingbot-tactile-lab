"""Expert overlay and paired-data validation tests, no GPU or SAPIEN import."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
SOURCE=Path(os.environ.get("ROBOTWIN_SOURCE", "/mnt/relaydata/lingbot/RoboTwin"))
from force_probe import ForceProbe
from make_expert_overlay import patched_source
from make_overlay import create_overlay
from validate_expert_data import validate
from build_paired_index import build


class ExpertTests(unittest.TestCase):
    def test_exact_expert_step_anchors_and_save_marker(self):
        source=(SOURCE/'envs/_base_task.py').read_text()
        patched=patched_source(source)
        self.assertEqual(source.count('self.scene.step()'),patched.count('self.scene.step()'))
        self.assertEqual(source.count('if self.check_success():'),patched.count('if self.check_success():'))
        self.assertIn('self._force_probe.begin_motion("expert_dense")',patched)
        self.assertIn('self._force_probe.begin_motion("expert_together")',patched)
        self.assertEqual(patched.count('self._force_probe.sample('),3)
        self.assertIn('self._force_probe.record_saved_frame(self.FRAME_IDX, pkl_dic, self.robot)',patched)
        self.assertIn('pkl_dic.pop("tactile_force", None)',patched)
        self.assertLess(patched.index('is_stable, unstable_list = self.check_stable()'),
                        patched.index('if self.save_data and os.environ.get("FORCE_PROBE_EXPERT_DIR")'))

    def test_overlay_builder_uses_expert_transform_not_basic(self):
        source_text=(SOURCE/'envs/_base_task.py').read_text()
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'src'; (src/'envs').mkdir(parents=True)
            (src/'envs/_base_task.py').write_text(source_text)
            dest=Path(td)/'overlay'
            create_overlay(src,dest,transform=patched_source)
            actual=(dest/'envs/_base_task.py').read_text()
            self.assertIn('record_saved_frame(self.FRAME_IDX, pkl_dic, self.robot)',actual)
            self.assertEqual(actual.count('self._force_probe.sample('),3)

    def test_saved_frames_join_hdf5_and_complete_manifest(self):
        import numpy as np
        import h5py
        class Joint:
            def __init__(self,name): self.name=name
            def get_name(self): return self.name
            @property
            def child_link(self): return NS(get_name=lambda:self.name.replace('joint','link'))
        class Entity:
            def get_active_joints(self): return [Joint('fl_joint7')]
            def get_qpos(self): return np.array([.1])
            def get_qvel(self): return np.array([0.])
            def get_links(self): return [NS(get_name=lambda:'fl_link7')]
        robot=NS(left_gripper=[(Joint('fl_joint7'),1,0)],right_gripper=[(Joint('fr_joint7'),1,0)],
                 left_fix_gripper_name=[],right_fix_gripper_name=[],left_entity=Entity(),right_entity=Entity())
        scene=NS(get_contacts=lambda:[])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            probe=ForceProbe(root,task='lift_pot',seed=17,episode=0,dt=.004,robot=robot)
            obs={'joint_action':{'vector':np.arange(14,dtype=np.float32)},
                 'observation':{'head_camera':{'rgb':NS(shape=(240,320,3))}}}
            probe.record_observation(obs,0)
            probe.record_saved_frame(0,obs,robot)
            probe.begin_motion('expert_dense')
            probe.sample(scene,robot,1)
            probe.record_observation(obs,0)
            probe.record_saved_frame(1,obs,robot)
            raw=probe.raw_path
            self.assertFalse(probe.failed)
            self.assertEqual(probe.saved_frames,2)
            probe.close()
            hf=root/'episode0.hdf5'
            with h5py.File(hf,'w') as f:
                f.create_dataset('joint_action/vector',data=np.stack([np.arange(14),np.arange(14)]))
                for cam in ('head_camera','left_camera','right_camera'):
                    f.create_dataset(f'observation/{cam}/rgb',data=np.array([1,2]))
            o=raw.with_suffix('.obs.jsonl'); frames=raw.with_suffix('.frames.jsonl')
            result=validate(raw,o,frames,hf)
            self.assertEqual(result['physics_steps'],1)
            self.assertEqual(result['saved_frames'],2)
            self.assertEqual([r['physics_step'] for r in [json.loads(x) for x in frames.read_text().splitlines()][1:]],[0,1])
            completion=root/'completion.json'
            completion.write_text(json.dumps({'schema':'phizju_expert_episode_complete_v1',
                'seed':17,'episode':0,'success':True,'sha256':result['sha256']}))
            self.assertEqual(validate(raw,o,frames,hf,completion)['saved_frames'],2)
            paired=list(build(raw,o,frames,hf,completion))
            self.assertEqual(len(paired),1)
            self.assertEqual(paired[0]['input']['frame_index'],0)
            self.assertFalse(paired[0]['input']['tactile_window_valid'])
            self.assertEqual(paired[0]['target']['action_source_frame_index'],1)
            self.assertNotIn('target',paired[0]['input'])
            with h5py.File(hf,'r+') as f:
                f['joint_action/vector'][1,0]=77
            with self.assertRaisesRegex(ValueError,'HDF5 command state'):
                validate(raw,o,frames,hf)


if __name__=='__main__':unittest.main()

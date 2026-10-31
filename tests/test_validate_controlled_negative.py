"""A simulated miss cannot be promoted into the success-only expert registry."""
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from validate_controlled_negative import build_card,sha256


class ControlledNegativeCardTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name);self.trial=self.repo/'data/negative/open_grasp'
        (self.trial/'force').mkdir(parents=True)
        (self.trial/'demonstration/data').mkdir(parents=True)
        self.raw=self.trial/'force/trial.jsonl';self.obs=self.raw.with_suffix('.obs.jsonl');self.frames=self.raw.with_suffix('.frames.jsonl')
        self.hdf5=self.trial/'demonstration/data/episode0.hdf5'
        for p in (self.raw,self.obs,self.frames,self.hdf5):p.write_bytes(('private-test-'+p.name).encode())
        hashes={name:sha256(p) for name,p in [('raw',self.raw),('obs',self.obs),('frames',self.frames),('hdf5',self.hdf5)]}
        ident={'task':'pick_diverse_bottles','seed':100000,'episode':0,
               'source_machine_role':'relay_single_rtx3090','source_hostname':socket.gethostname(),
               'planner_ok':True,'goal_ok':False,'intervened_grasps':2,
               'accepted_for_expert_training':False}
        self.gate=dict(ident,schema='lingbot_controlled_open_grasp_gate_v1',
                       status='controlled_negative_not_training_approved',saved_frame_markers=2,
                       source_sha256={x:hashes[x] for x in ('raw','obs','frames')})
        self.gate_path=self.trial/'negative_gate.json';self.gate_path.write_text(json.dumps(self.gate))
        self.manifest=dict(ident,schema='lingbot_controlled_negative_complete_v1',
                           status='quarantined_negative_not_training_or_eval',
                           negative_gate_sha256=sha256(self.gate_path),
                           contact_physics_steps_by_side={'left':1,'right':0},
                           physics_steps=2,camera_observations=2,saved_frames=2,dt_s=.004,
                           saved_first_step=0,saved_last_step=2,sha256=hashes)
        (self.trial/'negative_completion.json').write_text(json.dumps(self.manifest))
        self.current={key:self.manifest[key] for key in ('task','seed','episode','physics_steps','camera_observations','saved_frames','dt_s','saved_first_step','saved_last_step','sha256')}
        self.physics=[{'schema':'header'},
                      {'physics_step':1,'left':{'points':1},'right':{'points':0}},
                      {'physics_step':2,'left':{'points':0},'right':{'points':0}}]

    def verify(self):
        with patch('validate_controlled_negative.validate',return_value=self.current), \
             patch('validate_controlled_negative.read_jsonl',return_value=self.physics):
            return build_card(self.repo,self.trial)

    def test_independent_negative_card_has_no_expert_approval(self):
        c=self.verify()
        self.assertTrue(c['validated_same_run_rgb_hdf5'])
        self.assertTrue(c['observed_any_contact'])
        self.assertFalse(c['accepted_for_expert_training'])
        self.assertEqual(c['status'],'quarantined_controlled_negative_not_training_or_eval')
        self.assertEqual(c['contact_physics_steps_by_side'],{'left':1,'right':0})

    def test_gate_tamper_and_positive_success_markers_rejected(self):
        self.gate['intervened_grasps']=1;self.gate_path.write_text(json.dumps(self.gate))
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.verify()
        self.gate['intervened_grasps']=2;self.gate_path.write_text(json.dumps(self.gate))
        self.manifest['negative_gate_sha256']=sha256(self.gate_path)
        (self.trial/'negative_completion.json').write_text(json.dumps(self.manifest))
        (self.trial/'completion.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'positive expert completion forbidden'):self.verify()


if __name__=='__main__':unittest.main()

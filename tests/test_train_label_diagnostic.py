"""No held-out target or future-force label enters training-fold diagnostic."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from train_label_diagnostic import diagnose


class FakeEpisode:
    def __init__(self,repo,trial,*,instruction):
        if trial not in ('a','b','c'):raise AssertionError('held-out command labels read')
        self.trial=trial;assert instruction=='Lift the pot with both arms'
    def __len__(self):return 2
    def sample(self,index,*,include_images):
        assert not include_images
        h=np.zeros((5,4,3),dtype=np.float32)
        if index==1:h[-1,0]=[1,2,1]
        return {'input':{'tactile_history':h,'tactile_mask':np.ones((5,4)),
                         'command_state_14':np.zeros(14,dtype=np.float32)},
                'target':{'action_command_14':np.ones(14,dtype=np.float32)*index},
                'metadata':{'next_frame_step_delta':15 if index==0 else 1}}


class FakeInvalidEpisode(FakeEpisode):
    def sample(self,index,*,include_images):
        row=super().sample(index,include_images=include_images)
        if index==1:row['metadata']['next_frame_step_delta']=0
        return row


class TrainLabelDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.repo=Path(self.temp.name);(self.repo/'datasets/cards').mkdir(parents=True)
        self.split={'status':'candidate_not_approved','evaluation_claims_approved':False,
                    'train':['a','b','c'],'validation':['heldout_val'],'test':['heldout_test']}
        self.path=self.repo/'datasets/split.json';self.path.write_text(json.dumps(self.split))

    def test_train_only_timesteps_and_contact_vs_noncontact_command_change(self):
        with patch('train_label_diagnostic.build_plan',return_value={'split':self.split}), \
             patch('train_label_diagnostic.PairedEpisode',FakeEpisode):
            result=diagnose(self.repo,self.path)
        self.assertEqual(result['frame_physics_step_delta_counts'],{'1':3,'15':3})
        self.assertEqual(result['by_trial_and_observed_past_contact']['a']['past_contact']['changed_command_pairs'],1)
        self.assertEqual(result['by_trial_and_observed_past_contact']['a']['no_past_contact']['changed_command_pairs'],0)
        self.assertEqual(result['by_trial_and_observed_past_contact']['a']['past_contact']['mean_abs_command_delta_14d'],1.)

    def test_reject_noncausal_and_modified_split(self):
        with patch('train_label_diagnostic.build_plan',return_value={'split':self.split}), \
             patch('train_label_diagnostic.PairedEpisode',FakeInvalidEpisode):
            with self.assertRaisesRegex(ValueError,'noncausal'):
                diagnose(self.repo,self.path)
        self.path.write_text(json.dumps(dict(self.split,status='approved')))
        with patch('train_label_diagnostic.build_plan',return_value={'split':self.split}):
            with self.assertRaisesRegex(ValueError,'mismatch'):
                diagnose(self.repo,self.path)


if __name__=='__main__':unittest.main()

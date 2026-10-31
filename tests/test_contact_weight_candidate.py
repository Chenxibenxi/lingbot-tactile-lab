"""Past-only, train-only seed-balanced candidate loss weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from contact_weight_candidate import build_weights


class FakeEpisode:
    sizes={'a':(1,2),'b':(2,2),'c':(1,1)}
    def __init__(self,repo,trial,*,instruction):
        if trial not in self.sizes:raise AssertionError('held-out episode sampled for training weight')
        self.trial=trial;assert instruction=='Lift the pot with both arms'
    def __len__(self):return sum(self.sizes[self.trial])
    def sample(self,index,*,include_images):
        assert not include_images
        history=np.zeros((5,4,3),dtype=np.float32)
        if index<self.sizes[self.trial][0]:history[-1,0]=[1.,2.,1.]
        return {'input':{'tactile_history':history,'tactile_mask':np.ones((5,4)),
                         'tactile_window_valid':True},
                'metadata':{'next_frame_step_delta':25}}


class MissingContactEpisode(FakeEpisode):
    sizes={'a':(0,2),'b':(2,2),'c':(1,1)}


class TrainWeightsTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.repo=Path(self.temp.name);(self.repo/'datasets/cards').mkdir(parents=True)
        self.split={'status':'candidate_not_approved','evaluation_claims_approved':False,
                    'train':['a','b','c'],'validation':['d'],'test':['e']}
        self.path=self.repo/'datasets/split.json';self.path.write_text(json.dumps(self.split))

    def test_balanced_seed_class_weights_and_no_heldout_samples(self):
        with patch('contact_weight_candidate.build_plan',return_value={'split':self.split}), \
             patch('contact_weight_candidate.PairedEpisode',FakeEpisode):
            candidate=build_weights(self.repo,self.path)
        self.assertFalse(candidate['training_approval'])
        self.assertEqual(candidate['n_train_pairs'],9)
        self.assertEqual(candidate['observed_past_contact_counts_by_trial']['a'],
                         {'past_contact':1,'no_past_contact':2})
        self.assertAlmostEqual(candidate['loss_weight_by_trial_and_past_contact']['a']['past_contact'],1.5)
        self.assertAlmostEqual(candidate['loss_weight_by_trial_and_past_contact']['a']['no_past_contact'],.75)
        for trial,counts in candidate['observed_past_contact_counts_by_trial'].items():
            weights=candidate['loss_weight_by_trial_and_past_contact'][trial]
            self.assertAlmostEqual(sum(counts[label]*weights[label] for label in counts),3.)

    def test_no_contact_or_modified_split_refused(self):
        with patch('contact_weight_candidate.build_plan',return_value={'split':self.split}), \
             patch('contact_weight_candidate.PairedEpisode',MissingContactEpisode):
            with self.assertRaisesRegex(ValueError,'both observed contact'):
                build_weights(self.repo,self.path)
        self.path.write_text(json.dumps(dict(self.split,status='approved')))
        with patch('contact_weight_candidate.build_plan',return_value={'split':self.split}):
            with self.assertRaisesRegex(ValueError,'mismatch'):
                build_weights(self.repo,self.path)


if __name__=='__main__':unittest.main()

"""Matched RGB/action ablations and causal-past-only negative control."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from paired_ablation import make_view,audit_pilot_views
from paired_samples import CAMERAS


def fixture(trial='trial_a'):
    history=np.zeros((5,4,3),dtype=np.float32)
    history[0,0]=[1,2,.5];history[-1,-1]=[8,9,1]
    return {'input':{'command_state_14':np.ones(14,dtype=np.float32),
                     'images_rgb':{cam:np.zeros((240,320,3),dtype=np.uint8) for cam in CAMERAS},
                     'tactile_history':history,'tactile_mask':np.ones((5,4),dtype=np.float32),
                     'tactile_window_valid':True,'tactile_age_ms':np.float32(0.),
                     'capture_physics_step':100,'instruction':'Lift the pot with both arms'},
            'target':{'action_command_14':np.arange(14,dtype=np.float32)},
            'metadata':{'trial_id':trial,'next_frame_step_delta':25}}


class FakeEpisode:
    def __init__(self,repo,trial,*,instruction):
        self.trial=trial
        assert instruction=='Lift the pot with both arms'
    def __len__(self):return 1
    def sample(self,index,*,include_images):
        assert index==0 and include_images
        return fixture(self.trial)


class PairedAblationTests(unittest.TestCase):
    def test_three_views_keep_the_same_rgb_state_label_and_no_future(self):
        sample=fixture();original=sample['input']['tactile_history'].copy()
        views=[make_view(sample,mode) for mode in ('visual_only','tactile','permuted_causal_history')]
        self.assertTrue(np.array_equal(sample['input']['tactile_history'],original))
        for view in views:
            self.assertIs(view['input']['images_rgb'],sample['input']['images_rgb'])
            self.assertTrue(np.array_equal(view['target']['action_command_14'],sample['target']['action_command_14']))
            self.assertEqual(view['metadata'],sample['metadata'])
            self.assertNotIn('action_command_14',view['input'])
        self.assertFalse(views[0]['input']['tactile_window_valid'])
        self.assertFalse(np.any(views[0]['input']['tactile_mask']))
        self.assertTrue(np.array_equal(views[2]['input']['tactile_history'],original[::-1,::-1]))
        self.assertNotIn('target',views[2]['input'])
        with self.assertRaisesRegex(ValueError,'future'):
            make_view({'input':dict(sample['input'],tactile_future=1),
                       'target':sample['target'],'metadata':sample['metadata']},'tactile')

    def test_full_pilot_audit_matching_and_tampered_split_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);(repo/'datasets/cards').mkdir(parents=True)
            split={'status':'candidate_not_approved','evaluation_claims_approved':False,
                   'train':['trial_a'],'validation':['trial_b'],'test':['trial_c']}
            path=repo/'datasets/split.json';path.write_text(json.dumps(split))
            with patch('paired_ablation.build_plan',return_value={'split':split}), \
                 patch('paired_ablation.PairedEpisode',FakeEpisode):
                result=audit_pilot_views(repo,path,instruction='Lift the pot with both arms')
                self.assertEqual([result['folds'][f]['matched_pairs'] for f in ('train','validation','test')],[1,1,1])
                self.assertEqual(result['folds']['test']['pairs_changed_by_permutation'],1)
                path.write_text(json.dumps(dict(split,evaluation_claims_approved=True)))
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    audit_pilot_views(repo,path,instruction='Lift the pot with both arms')


if __name__=='__main__':unittest.main()

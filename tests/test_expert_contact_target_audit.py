"""Expert future contact is label-only, with train-fold and clock gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from expert_contact_target_audit import audit_train


class FakeEpisode:
    def __init__(self,repo,trial,*,instruction):
        if trial not in ('a','b','c'):raise AssertionError('held-out source read')
        assert instruction=='Lift the pot with both arms'
        self.raw=Path('not-opened-directly.jsonl')
    def __len__(self):return 2
    def sample(self,index,*,include_images):
        assert not include_images
        h=np.zeros((5,4,3),dtype=np.float32)
        if index==1:h[-1,0]=[1,1,1];h[-1,2]=[1,1,1]
        return {'input':{'capture_physics_step':index,
                         'tactile_history':h,'tactile_mask':np.ones((5,4))},
                'metadata':{'next_frame_step_delta':1}}


def physics():
    return [{'physics_step':step,
             'left':{'points':1,'load_proxy_n':1.},
             'right':{'points':1,'load_proxy_n':1.}}
            for step in (1,2)]


class ExpertContactTargetAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.repo=Path(self.temp.name);(self.repo/'datasets/cards').mkdir(parents=True)
        self.split={'status':'candidate_not_approved','evaluation_claims_approved':False,
                    'train':['a','b','c'],'validation':['heldout_val'],'test':['heldout_test']}
        self.path=self.repo/'datasets/split.json';self.path.write_text(json.dumps(self.split))

    def run_audit(self,rows):
        with patch('expert_contact_target_audit.build_plan',return_value={'split':self.split}), \
             patch('expert_contact_target_audit.PairedEpisode',FakeEpisode), \
             patch('expert_contact_target_audit.read_jsonl',return_value=[{'dt_s':.004},*rows]):
            return audit_train(self.repo,self.path)

    def test_first_new_contact_is_future_label_only_not_current_input(self):
        out=self.run_audit(physics())
        self.assertEqual(out['status'],'TRAIN_ONLY_DIAGNOSTIC_NO_AUX_TRAINING_APPROVAL')
        for trial in self.split['train']:
            counts=out['by_trial'][trial]
            self.assertEqual(counts['n_pairs'],2)
            self.assertEqual(counts['left_future_any'],2)
            self.assertEqual(counts['left_new_onset_from_no_prior'],1)
            self.assertEqual(counts['both_new_onset_same_interval'],1)
        self.assertNotIn('labels',out)

    def test_missing_future_step_and_modified_split_refused(self):
        with self.assertRaisesRegex(ValueError,'missing physics'):
            self.run_audit(physics()[:1])
        self.path.write_text(json.dumps(dict(self.split,status='approved')))
        with self.assertRaisesRegex(ValueError,'mismatch'):
            self.run_audit(physics())


if __name__=='__main__':unittest.main()

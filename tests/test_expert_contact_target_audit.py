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

    def test_bottle_task_uses_its_own_split_and_explicit_language(self):
        class BottleEpisode(FakeEpisode):
            def __init__(self,repo,trial,*,instruction):
                self.raw=Path('not-opened-directly.jsonl')
                assert trial in ('a','b','c')
                assert instruction=='Pick and place the two diverse bottles'
        with patch('expert_contact_target_audit.build_plan',return_value={'split':self.split}) as plan, \
             patch('expert_contact_target_audit.PairedEpisode',BottleEpisode), \
             patch('expert_contact_target_audit.read_jsonl',return_value=[{'dt_s':.004},*physics()]):
            result=audit_train(self.repo,self.path,task='pick_diverse_bottles')
        self.assertEqual(plan.call_args.kwargs['task'],'pick_diverse_bottles')
        self.assertEqual(result['by_trial']['a']['left_new_onset_from_no_prior'],1)

    def test_unassigned_new_expert_cannot_reshuffle_frozen_v1(self):
        extra=self.repo/'datasets/cards/new_unassigned_expert.json';extra.write_text('{}')
        with patch('expert_contact_target_audit.build_plan',return_value={'split':self.split}) as plan, \
             patch('expert_contact_target_audit.PairedEpisode',FakeEpisode), \
             patch('expert_contact_target_audit.read_jsonl',return_value=[{'dt_s':.004},*physics()]):
            audit_train(self.repo,self.path)
        selected=plan.call_args.args[1]
        self.assertNotIn(extra,selected)
        self.assertEqual(len(selected),5)

    def test_bottle_v2_verifies_its_separate_frozen_train_and_no_old_recompute(self):
        v2=self.repo/'datasets/relay_pickdiverse_seed_split_candidate_v2.json'
        v2.write_text(json.dumps(self.split))
        with patch('extend_relay_bottle_split.build',return_value={'split':self.split}) as rebuilt, \
             patch('expert_contact_target_audit.build_plan',side_effect=AssertionError('v2 cannot reshuffle')), \
             patch('expert_contact_target_audit.PairedEpisode',FakeEpisode), \
             patch('expert_contact_target_audit.read_jsonl',return_value=[{'dt_s':.004},*physics()]):
            result=audit_train(self.repo,v2,task='pick_diverse_bottles',
                               instruction='Lift the pot with both arms')
        self.assertEqual(rebuilt.call_count,1)
        self.assertEqual(result['by_trial']['b']['left_new_onset_from_no_prior'],1)


if __name__=='__main__':unittest.main()

"""Prediction coverage, provenance, and contact-stratified macro-seed loss."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from contact_stratified_eval import evaluate,MODES


class FakeEpisode:
    def __init__(self,repo,trial,*,instruction):
        self.trial=trial;assert instruction=='Lift the pot with both arms'
    def __len__(self):return 2
    def sample(self,index,*,include_images):
        assert not include_images
        h=np.zeros((5,4,3),dtype=np.float32)
        if index==1:h[-1,0]=[2,3,1]
        return {'input':{'tactile_history':h,'tactile_mask':np.ones((5,4),dtype=np.float32)},
                'target':{'action_command_14':np.arange(14,dtype=np.float32)},
                'metadata':{'frame_index':index}}


class ContactStratifiedEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name);(self.repo/'datasets/cards').mkdir(parents=True)
        self.split={'status':'candidate_not_approved','evaluation_claims_approved':False,
                    'train':['a','b','c'],'validation':['d'],'test':['e']}
        self.path=self.repo/'datasets/split.json';self.path.write_text(json.dumps(self.split))
        self.manifest={'schema':'lingbot_tactile_predictions_v1',
                       'split_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                       'vision_checkpoint_sha256':'a'*64,
                       'model_adapter_sha256_by_mode':{mode:'b'*64 for mode in MODES},
                       'rows':[]}
        for trial in 'abcde':
            for frame in (0,1):
                for mode,delta in [('visual_only',1),('tactile',0),('permuted_causal_history',2)]:
                    self.manifest['rows'].append({'trial_id':trial,'frame_index':frame,
                        'mode':mode,'action_14':(np.arange(14)+delta).tolist()})

    def assess(self,manifest):
        with patch('contact_stratified_eval.build_plan',return_value={'split':self.split}), \
             patch('contact_stratified_eval.PairedEpisode',FakeEpisode):
            return evaluate(self.repo,self.path,manifest)

    def test_macroseed_contact_strata_and_labels_come_only_from_source(self):
        result=self.assess(self.manifest)
        self.assertEqual(result['status'],'offline_imitation_error_only_not_task_success')
        self.assertEqual(result['metrics']['train']['past_contact']['tactile']['n_pairs'],3)
        self.assertEqual(result['metrics']['train']['no_past_contact']['tactile']['n_pairs'],3)
        self.assertEqual(result['metrics']['validation']['past_contact']['visual_only']['independent_seeds_with_samples'],1)
        self.assertEqual(result['metrics']['test']['past_contact']['visual_only']['macro_seed_mae_14d'],1)
        self.assertEqual(result['metrics']['test']['past_contact']['tactile']['macro_seed_mae_14d'],0)
        self.assertEqual(result['metrics']['test']['past_contact']['permuted_causal_history']['macro_seed_joint_mae_12d'],2)

    def test_new_unassigned_card_does_not_reassign_frozen_v1_scorer(self):
        extra=self.repo/'datasets/cards/unassigned_expert.json';extra.write_text('{}')
        with patch('contact_stratified_eval.build_plan',return_value={'split':self.split}) as plan, \
             patch('contact_stratified_eval.PairedEpisode',FakeEpisode):
            score=evaluate(self.repo,self.path,self.manifest)
        self.assertEqual(score['task'],'lift_pot')
        self.assertFalse(score['model_artifacts_revalidated'])
        self.assertNotIn(extra,plan.call_args.args[1])
        self.assertEqual(len(plan.call_args.args[1]),5)

    def test_bottle_v2_requires_parent_verified_frozen_split_and_correct_instruction(self):
        path=self.repo/'datasets/relay_pickdiverse_seed_split_candidate_v2.json'
        path.write_text(json.dumps(self.split))
        predictions=dict(self.manifest,split_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        class BottleEpisode(FakeEpisode):
            def __init__(self,repo,trial,*,instruction):
                self.trial=trial
                assert instruction=='Pick and place the two diverse bottles'
        with patch('extend_relay_bottle_split.build',return_value={'split':self.split}) as rebuilt, \
             patch('contact_stratified_eval.build_plan',side_effect=AssertionError('v2 must not reshuffle')), \
             patch('contact_stratified_eval.PairedEpisode',BottleEpisode):
            score=evaluate(self.repo,path,predictions,task='pick_diverse_bottles')
            self.assertEqual(score['task'],'pick_diverse_bottles')
            self.assertEqual(rebuilt.call_count,1)
            with self.assertRaisesRegex(ValueError,'v2 split only supports bottle'):
                evaluate(self.repo,path,predictions,task='lift_pot')
            path.write_text(json.dumps(dict(self.split,test=['a'])))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                evaluate(self.repo,path,predictions,task='pick_diverse_bottles')

    def test_missing_duplicate_nonfinite_or_injected_target_fail(self):
        with self.assertRaisesRegex(ValueError,'incomplete matched'):
            self.assess(dict(self.manifest,rows=self.manifest['rows'][:-1]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.assess(dict(self.manifest,rows=[*self.manifest['rows'],self.manifest['rows'][0]]))
        bad=dict(self.manifest['rows'][0],target_action_14=[0]*14)
        with self.assertRaisesRegex(ValueError,'unknown'):
            self.assess(dict(self.manifest,rows=[bad,*self.manifest['rows'][1:]]))
        bad=dict(self.manifest['rows'][0],action_14=[float('nan')]*14)
        with self.assertRaisesRegex(ValueError,'finite'):
            self.assess(dict(self.manifest,rows=[bad,*self.manifest['rows'][1:]]))
        with self.assertRaisesRegex(ValueError,'pinned split'):
            self.assess(dict(self.manifest,split_sha256='0'*64))


if __name__=='__main__':unittest.main()

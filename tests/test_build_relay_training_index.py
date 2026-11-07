"""Candidate dataset index cannot launder negative trials or heldout seeds."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from build_relay_training_index import build


class PairedIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)
        for name in ('datasets/cards','configs','data/paired'):(self.repo/name).mkdir(parents=True)
        (self.repo/'datasets/provenance_v1.json').write_text(json.dumps({
            'schema':'lingbot_tactile_dataset_registry_v1','approved_paired_episodes':[]}))
        self.trials={f'relay_pickdiverse_s{s}_expert0':s for s in (1,2,3,4,5)}
        for trial,seed in self.trials.items():
            (self.repo/'data/paired'/trial).mkdir()
            (self.repo/'datasets/cards'/f'{trial}.json').write_text(json.dumps({
                'task':'pick_diverse_bottles','trial_id':trial,'seed':seed,'physics_steps':30}))
        self.split={'schema':'lingbot_tactile_seed_split_plan_v1',
            'status':'candidate_not_approved','source_machine_role':'relay_single_rtx3090',
            'evaluation_claims_approved':False,
            'train':[f'relay_pickdiverse_s{s}_expert0' for s in (1,2,3)],
            'validation':['relay_pickdiverse_s4_expert0'],'test':['relay_pickdiverse_s5_expert0'],
            'task_seed_groups':{'pick_diverse_bottles':{
                'train':[1,2,3],'validation':[4],'test':[5]}},
            'expert_card_sha256':{trial:hashlib.sha256((self.repo/'datasets/cards'/f'{trial}.json').read_bytes()).hexdigest()
                                  for trial in self.trials}}
        self.split_path=self.repo/'datasets/relay_pickdiverse_seed_split_candidate_v1.json'
        self.norm_path=self.repo/'configs/relay_pickdiverse_train_norm_candidate_v1.json'
        self.save()

    def save(self):
        self.split_path.write_text(json.dumps(self.split))
        self.norm_path.write_text(json.dumps({'schema':'lingbot_tactile_train_norm_v1',
            'training_approval':False,'source_trials':self.split['train'],
            'split_candidate_sha256':hashlib.sha256(json.dumps(self.split,sort_keys=True).encode()).hexdigest()}))

    def mocked_build(self):
        repo=self.repo
        class Episode:
            def __init__(self,repo_arg,trial_id,*,instruction):
                self.card=json.loads((repo/'datasets/cards'/f'{trial_id}.json').read_text())
            def __len__(self):return 1
            def sample(self,index,*,include_images=True):
                assert include_images and index==0
                seed=self.card['seed'];trial=self.card['trial_id']
                history=np.zeros((5,4,3),dtype=np.float32)
                if seed==1:history[0,0,0]=1.
                return {'metadata':{'task':'pick_diverse_bottles','seed':seed,'trial_id':trial,
                                    'frame_index':1,'target_frame_index':2,'next_frame_step_delta':5},
                        'input':{'capture_physics_step':10,'tactile_window_valid':True,
                                 'tactile_history':history,'tactile_mask':np.ones((5,4)),
                                 'images_rgb':{c:np.zeros((240,320,3),dtype=np.uint8)
                                               for c in ('head_camera','left_camera','right_camera')}}}
        return Episode

    def test_exact_same_run_fold_index_and_contact_aggregation(self):
        with patch('build_relay_training_index.TASKS',{'pick_diverse_bottles':(
                self.split_path.name,self.norm_path.name,'Pick bottles')}), \
             patch('build_relay_training_index.PairedEpisode',self.mocked_build()), \
             patch('build_relay_training_index.rebuild_expert_card',
                   side_effect=lambda trial,repo:json.loads((self.repo/'datasets/cards'/f'{trial.name}.json').read_text())):
            rows,card=build(self.repo)
            self.assertEqual(len(rows),5)
            self.assertEqual(card['tasks']['pick_diverse_bottles']['folds']['train']['pairs'],3)
            self.assertEqual(card['tasks']['pick_diverse_bottles']['folds']['train']['pairs_with_observed_contact'],1)
            self.assertFalse(card['training_approval'])
            self.assertEqual(card['negative_or_self_rollout_pairs'],0)
            self.split['task_seed_groups']['pick_diverse_bottles']['validation']=[1]
            self.save()
            with self.assertRaisesRegex(ValueError,'leaks'):
                build(self.repo)

    def test_bottle_v2_only_adds_train_and_freezes_v1_heldout(self):
        new_ids=['relay_pickdiverse_s100013_onsetpilot_expert0',
                 'relay_pickdiverse_s100015_onsetpilot_expert0']
        for trial,seed in zip(new_ids,(100013,100015)):
            (self.repo/'data/paired'/trial).mkdir()
            (self.repo/'datasets/cards'/f'{trial}.json').write_text(json.dumps({
                'task':'pick_diverse_bottles','trial_id':trial,'seed':seed,'physics_steps':30}))
        v2=json.loads(self.split_path.read_text())
        v2['parent_v1_split_sha256']=hashlib.sha256(self.split_path.read_bytes()).hexdigest()
        v2['new_unassigned_parent_experts']=new_ids
        v2['train']=sorted(v2['train']+new_ids)
        v2['task_seed_groups']['pick_diverse_bottles']['train']=[1,2,3,100013,100015]
        for trial in new_ids:
            v2['expert_card_sha256'][trial]=hashlib.sha256((self.repo/'datasets/cards'/f'{trial}.json').read_bytes()).hexdigest()
        v2_path=self.repo/'datasets/relay_pickdiverse_seed_split_candidate_v2.json'
        v2_path.write_text(json.dumps(v2))
        (self.repo/'configs/relay_pickdiverse_train_norm_candidate_v2.json').write_text(json.dumps({
            'schema':'lingbot_tactile_train_norm_v1','training_approval':False,
            'source_trials':v2['train'],
            'split_candidate_sha256':hashlib.sha256(json.dumps(v2,sort_keys=True).encode()).hexdigest()}))
        with patch('build_relay_training_index.TASKS',{'pick_diverse_bottles':(
                self.split_path.name,self.norm_path.name,'Pick bottles')}), \
             patch('build_relay_training_index.PairedEpisode',self.mocked_build()), \
             patch('build_relay_training_index.rebuild_expert_card',
                   side_effect=lambda trial,repo:json.loads((self.repo/'datasets/cards'/f'{trial.name}.json').read_text())):
            rows,card=build(self.repo,bottle_v2=True)
            self.assertEqual(len(rows),7)
            self.assertEqual(card['tasks']['pick_diverse_bottles']['folds']['train']['pairs'],5)
            self.assertEqual({x['trial_id'] for x in rows if x['fold']=='validation'},set(self.split['validation']))
            self.assertEqual({x['trial_id'] for x in rows if x['fold']=='test'},set(self.split['test']))
            self.assertEqual(card['variant'],'bottle_v2_add_train_keep_v1_heldout')
            self.assertFalse(card['training_approval'])
            v2['test']=new_ids[:1];v2_path.write_text(json.dumps(v2))
            with self.assertRaisesRegex(ValueError,'preserve old heldout'):
                build(self.repo,bottle_v2=True)

    def test_negative_and_card_sha_tamper_are_refused(self):
        with patch('build_relay_training_index.TASKS',{'pick_diverse_bottles':(
                self.split_path.name,self.norm_path.name,'Pick bottles')}), \
             patch('build_relay_training_index.PairedEpisode',self.mocked_build()), \
             patch('build_relay_training_index.rebuild_expert_card',
                   side_effect=lambda trial,repo:json.loads((self.repo/'datasets/cards'/f'{trial.name}.json').read_text())):
            (self.repo/'datasets/cards/relay_pickdiverse_s1_expert0.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'expert source'):
                build(self.repo)
            self.split['train'][0]='relay_pickdiverse_s1_open_grasp_spatial1';self.save()
            with self.assertRaisesRegex(ValueError,'negative'):
                build(self.repo)
            self.split['train'][0]='../_expert';self.save()
            with self.assertRaisesRegex(ValueError,'malformed'):
                build(self.repo)


if __name__=='__main__':unittest.main()

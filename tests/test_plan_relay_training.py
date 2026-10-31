"""Technical split and train-fold normalization cannot read held-out seeds."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from plan_relay_training import build_plan


class FakeEpisode:
    def __init__(self,repo,trial,*,instruction):
        self.trial=trial
        assert instruction=='Lift the pot with both arms'
    def __len__(self):return 1
    def sample(self,index,*,include_images):
        assert index==0 and not include_images
        seed=int(self.trial.split('_')[-1])
        history=np.ones((5,4,3),dtype=np.float32)*seed
        history[...,2]=.5
        return {'metadata':{'trial_id':self.trial},'input':{
            'tactile_window_valid':True,
            'tactile_history':history,
            'tactile_mask':np.ones((5,4),dtype=np.float32)}}


class RelaySeedPlanTests(unittest.TestCase):
    def test_minimum_five_seed_candidate_and_train_only_norm(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);cards_dir=repo/'datasets/cards';cards_dir.mkdir(parents=True)
            cards=[];paths=[]
            for seed in range(1,6):
                card={'schema':'lingbot_tactile_dataset_card_v1','status':'candidate_unassigned_split',
                      'source_machine_role':'relay_single_rtx3090','source_hostname':'admin001-MS-7D90',
                      'task':'lift_pot','seed':seed,'trial_id':f'run_{seed}',
                      'data_path_relative':f'data/paired/run_{seed}'}
                p=cards_dir/f'run_{seed}.json';p.write_text(json.dumps(card));cards.append(card);paths.append(p)
            card_by_name={c['trial_id']:c for c in cards}
            with patch('plan_relay_training.build_card',side_effect=lambda path,repo:card_by_name[Path(path).name]), \
                 patch('plan_relay_training.PairedEpisode',FakeEpisode):
                with self.assertRaisesRegex(ValueError,'distinct seeds'):
                    build_plan(repo,paths[:4],task='lift_pot',instruction='Lift the pot with both arms')
                result=build_plan(repo,paths,task='lift_pot',instruction='Lift the pot with both arms')
            split=result['split'];norm=result['norm']
            self.assertEqual((len(split['train']),len(split['validation']),len(split['test'])),(3,1,1))
            self.assertEqual(set(norm['source_trials']),set(split['train']))
            self.assertFalse(norm['training_approval'])
            self.assertFalse(result['contact_norm']['training_approval'])
            self.assertEqual(set(result['contact_norm']['source_trials']),set(split['train']))
            self.assertEqual(result['contact_norm']['split_candidate_sha256'],norm['split_candidate_sha256'])
            self.assertFalse(split['evaluation_claims_approved'])
            self.assertEqual(sum(result['train_pairs_by_trial'].values()),3)
            self.assertEqual(len(split['expert_card_sha256']),5)

    def test_external_or_nonexpert_card_rejected_before_split(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);outside=repo/'external.json';outside.write_text('{}')
            with self.assertRaisesRegex(ValueError,'outside relay expert registry'):
                build_plan(repo,[outside],task='lift_pot',instruction='Lift the pot with both arms')


if __name__=='__main__':unittest.main()

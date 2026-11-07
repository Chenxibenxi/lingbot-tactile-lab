"""Do not cherry-pick attempted seeds or silently modify the old 3/1/1 folds."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from audit_bottle_onset_expansion import audit


class BottleExpansionAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)
        for folder in ('datasets/cards','data/paired'):(self.repo/folder).mkdir(parents=True)
        (self.repo/'datasets/provenance_v1.json').write_text(json.dumps({'approved_paired_episodes':[]}))
        self.split={'status':'candidate_not_approved','evaluation_claims_approved':False,
                    'train':['old_expert_train'],'validation':['old_expert_val'],'test':['old_expert_test'],
                    'task_seed_groups':{'pick_diverse_bottles':{
                        'train':[100000],'validation':[100012],'test':[100009]}}}
        self.split_path=self.repo/'datasets/relay_pickdiverse_seed_split_candidate_v1.json'
        self.split_path.write_text(json.dumps(self.split))
        self.card={'task':'pick_diverse_bottles','seed':100013,'status':'candidate_unassigned_split',
                   'saved_frames':100}
        (self.repo/'datasets/cards/relay_pickdiverse_s100013_onsetpilot_expert0.json').write_text(json.dumps(self.card))
        (self.repo/'data/paired/relay_pickdiverse_s100013_onsetpilot_expert0.status').write_text('exit_code=0\n')
        (self.repo/'data/paired/relay_pickdiverse_s100015_onsetpilot_expert0.status').write_text('exit_code=1\n')

    def run_audit(self):
        fake=type('Episode',(),{'card':self.card})
        with patch('audit_bottle_onset_expansion.rebuild_expert_card',return_value=self.card), \
             patch('audit_bottle_onset_expansion.PairedEpisode',return_value=fake), \
             patch('audit_bottle_onset_expansion.onset_counts',return_value={'causal_pairs':98,
                 'pairs_with_past_contact':44,'future_new_onset_intervals_by_side':{'left':1,'right':1}}):
            return audit(self.repo)

    def test_all_fixed_attempts_recorded_and_new_expert_stays_unassigned(self):
        result=self.run_audit()
        self.assertEqual(result['predeclared_seeds'],[100013,100015])
        self.assertEqual(result['successful_new_expert_candidates'],1)
        self.assertEqual(result['new_causal_pairs_unassigned'],98)
        self.assertFalse(result['training_approval'])
        self.assertIn('isolated',result['attempts'][1]['status'])

    def test_existing_split_cannot_steal_new_seed_or_trial(self):
        self.split['task_seed_groups']['pick_diverse_bottles']['train'].append(100013)
        self.split_path.write_text(json.dumps(self.split))
        with self.assertRaisesRegex(ValueError,'frozen old expert fold'):self.run_audit()
        self.split['task_seed_groups']['pick_diverse_bottles']['train']=[100000]
        self.split['test'].append('relay_pickdiverse_s100013_onsetpilot_expert0')
        self.split_path.write_text(json.dumps(self.split))
        with self.assertRaisesRegex(ValueError,'frozen old expert fold'):self.run_audit()


if __name__=='__main__':unittest.main()

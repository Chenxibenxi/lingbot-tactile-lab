"""New successes may join train candidate only; original heldout seeds stay frozen."""
from copy import deepcopy
import unittest
from extend_relay_bottle_split import NEW_IDS,EXPECTED_SEEDS,extend_preserving_heldout


class BottlesSplitV2Tests(unittest.TestCase):
    def setUp(self):
        self.old={'schema':'lingbot_tactile_seed_split_plan_v1','status':'candidate_not_approved',
                  'source_machine_role':'relay_single_rtx3090','evaluation_claims_approved':False,
                  'train':['old_train1','old_train2','old_train3'],
                  'validation':['old_val'],'test':['old_test'],
                  'expert_card_sha256':{name:'a'*64 for name in
                       ('old_train1','old_train2','old_train3','old_val','old_test')},
                  'task_seed_groups':{'pick_diverse_bottles':{
                      'train':[100000,100002,100007],
                      'validation':[100012],'test':[100009]}}}
        self.new=[{'schema':'lingbot_tactile_dataset_card_v1','task':'pick_diverse_bottles',
                   'status':'candidate_unassigned_split','source_machine_role':'relay_single_rtx3090',
                   'trial_id':trial,'seed':seed}
                  for trial,seed in zip(NEW_IDS,EXPECTED_SEEDS)]
        self.hashes={trial:'b'*64 for trial in NEW_IDS}

    def test_only_appends_new_train_seed_and_does_not_rewrite_v1(self):
        before=deepcopy(self.old)
        v2=extend_preserving_heldout(self.old,self.new,self.hashes,parent_sha='c'*64)
        self.assertEqual(self.old,before)
        self.assertEqual(v2['train'],sorted(before['train']+list(NEW_IDS)))
        self.assertEqual(v2['validation'],['old_val'])
        self.assertEqual(v2['test'],['old_test'])
        self.assertEqual(set(v2['task_seed_groups']['pick_diverse_bottles']['train']),
                         {100000,100002,100007,100013,100015})
        self.assertEqual(v2['parent_v1_split_sha256'],'c'*64)
        self.assertFalse(v2['evaluation_claims_approved'])

    def test_negative_role_duplicate_seed_or_old_fold_mutation_rejected(self):
        malformed=deepcopy(self.new);malformed[0]['status']='quarantined_negative'
        with self.assertRaisesRegex(ValueError,'not a verified'):
            extend_preserving_heldout(self.old,malformed,self.hashes,parent_sha='c'*64)
        old=deepcopy(self.old);old['task_seed_groups']['pick_diverse_bottles']['validation']=[100013]
        with self.assertRaisesRegex(ValueError,'crosses'):
            extend_preserving_heldout(old,self.new,self.hashes,parent_sha='c'*64)
        old=deepcopy(self.old);old['test'].append('moved_heldout')
        with self.assertRaisesRegex(ValueError,'not trusted'):
            extend_preserving_heldout(old,self.new,self.hashes,parent_sha='c'*64)


if __name__=='__main__':unittest.main()

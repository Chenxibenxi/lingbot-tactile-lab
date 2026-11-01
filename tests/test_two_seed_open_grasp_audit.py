"""A controlled failure does not authorize heldout leakage or generalization claims."""
import unittest
from two_seed_open_grasp_audit import relative_contact_change_percent,require_train_only_seeds


class TwoSeedAuditTests(unittest.TestCase):
    def test_descriptive_percent_drop_may_be_negative_without_cherry_picking(self):
        self.assertEqual(relative_contact_change_percent(829,61),92.642)
        self.assertEqual(relative_contact_change_percent(10,12),-20.)
        for a,b in ((0,0),(-1,0),(10,-1)):
            with self.assertRaises(ValueError):relative_contact_change_percent(a,b)

    def test_failure_seed_never_comes_from_heldout_or_duplicate(self):
        split={'schema':'lingbot_tactile_seed_split_plan_v1','status':'candidate_not_approved',
               'task_seed_groups':{'pick_diverse_bottles':{
                   'train':[100000,100002,100007],'validation':[100012],'test':[100009]}},
               'evaluation_claims_approved':False}
        require_train_only_seeds(split,[100000,100007])
        for chosen in ([100000,100009],[100000,100000],[100000,100012]):
            with self.assertRaises(ValueError):require_train_only_seeds(split,chosen)
        split['evaluation_claims_approved']=True
        with self.assertRaisesRegex(ValueError,'approval'):
            require_train_only_seeds(split,[100000,100007])


if __name__=='__main__':unittest.main()

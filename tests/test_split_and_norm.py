import unittest
import numpy as np
from split_and_norm import make_seed_splits, fit_train_only_stats


def card(task, seed, trial):
    return {'schema':'lingbot_tactile_dataset_card_v1',
            'source_machine_role':'relay_single_rtx3090',
            'source_hostname':'admin001-MS-7D90',
            'task':task,'seed':seed,'trial_id':trial}


def example(trial, force):
    h=np.zeros((5,4,3),dtype=np.float32);h[:,:,:2]=force;h[:,:,2]=1.
    return {'input':{'tactile_history':h,'tactile_mask':np.ones((5,4)),
                     'tactile_window_valid':True},'metadata':{'trial_id':trial}}


class SplitAndNormTests(unittest.TestCase):
    def test_same_seed_retries_stay_in_one_split(self):
        cards=[card(task,seed,f'{task}_{seed}') for task in ('lift_pot','click_bell') for seed in range(5)]
        cards.append(card('lift_pot',0,'lift_pot_0_retry'))
        plan=make_seed_splits(cards)
        self.assertEqual(plan, make_seed_splits(reversed(cards)))
        for task in ('lift_pot','click_bell'):
            groups=plan['task_seed_groups'][task]
            self.assertEqual(sum(len(x) for x in groups.values()),5)
            self.assertEqual(len(set().union(*(set(x) for x in groups.values()))),5)
        belonging=[name for name in ('train','validation','test') if 'lift_pot_0' in plan[name]]
        self.assertEqual(len(belonging),1)
        self.assertIn('lift_pot_0_retry',plan[belonging[0]])
        self.assertEqual(plan['status'],'candidate_not_approved')

    def test_fail_closed_on_low_seed_count_and_external_card(self):
        with self.assertRaisesRegex(ValueError,'distinct seeds'):
            make_seed_splits([card('lift_pot',s,f'trial{s}') for s in range(3)])
        external=card('lift_pot',0,'trial0');external['source_machine_role']='historic_lxc_two_rtx3090'
        with self.assertRaisesRegex(ValueError,'non-relay'):
            make_seed_splits([external])

    def test_val_outlier_cannot_affect_train_only_median(self):
        baseline=fit_train_only_stats([example('train_a',2),example('train_b',4)],{'train_a','train_b'})
        with_val=fit_train_only_stats([example('val',1e8),example('train_a',2),example('train_b',4)],{'train_a','train_b'})
        self.assertEqual(baseline,with_val)
        self.assertEqual(baseline['source_trials'],['train_a','train_b'])
        self.assertEqual(baseline['n_bins_by_finger'],[10]*4)
        with self.assertRaisesRegex(ValueError,'not every requested train trial'):
            fit_train_only_stats([example('train_a',2)],{'train_a','train_missing'})


if __name__=='__main__':unittest.main()

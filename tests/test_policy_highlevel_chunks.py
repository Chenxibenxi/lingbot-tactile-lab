"""H50 variable-duration expert take_action candidate contract; synthetic only."""
import unittest
from policy_highlevel_chunks import METHOD,ROLE,SOURCE_SHA,select_policy_h50

TRIAL='relay_liftpot_new_controller_seed100011_expert0'
COMPLETION='1'*64
RGB='2'*64


def rows(n=51):
    answer=[];step=30
    for i in range(n):
        duration=(17,86,181)[i%3];after=step+duration
        cmd=[float(i)/100]*14;cmd[6]=0.5;cmd[13]=0.5
        answer.append({'trial_id':TRIAL,'completion_sha256':COMPLETION,
                       'source_model_role':ROLE,'controller_method':METHOD,
                       'original_base_task_sha256':SOURCE_SHA,'action_index':i,
                       'before_physics_step':step,'after_physics_step':after,
                       'command_state_14':cmd})
        step=after
    return answer


def anchor():
    return {'trial_id':TRIAL,'completion_sha256':COMPLETION,
            'source_model_role':ROLE,'controller_method':METHOD,
            'original_base_task_sha256':SOURCE_SHA,'current_rgb_hdf5_sha256':RGB,
            'first_action_index':0,'capture_physics_step':30,
            'past_tactile_max_physics_step':30,'tactile_window_valid':True}


def gate(data=None,a=None,fold='train'):
    return select_policy_h50(rows() if data is None else data,
                             anchor() if a is None else a,
                             expected_trial=TRIAL,expected_completion_sha=COMPLETION,
                             source_physics_steps=6000,fold=fold)


class PolicyControllerActionClockTests(unittest.TestCase):
    def test_real_action_intervals_may_vary_without_inventing_uniform_fps(self):
        candidate=gate()
        self.assertEqual(candidate['target_action_count'],50)
        self.assertEqual({x['after_physics_step']-x['before_physics_step']
                          for x in candidate['targets']},{17,86,181})
        self.assertEqual(candidate['target_semantics'],'actual_take_action_calls_not_resampled_drive_ticks')
        self.assertFalse(candidate['training_approval'])
        self.assertFalse(candidate['test_fold_read'])

    def test_rule_expert_dense_ticks_or_policy_self_rollout_are_not_expert_labels(self):
        for role in ('expert_dense_post_scene_step','executed_self_rollout_not_expert_or_training_approved'):
            data=rows();data[0]['source_model_role']=role
            with self.subTest(role=role),self.assertRaisesRegex(ValueError,'source invalid'):
                gate(data)
        a=anchor();a['controller_method']='Base_Task.take_dense_action'
        with self.assertRaisesRegex(ValueError,'not bound'):gate(a=a)

    def test_future_force_wrong_camera_step_or_forged_source_hash_refused(self):
        for key,value in [('past_tactile_max_physics_step',31),
                          ('capture_physics_step',31),
                          ('original_base_task_sha256','0'*64),
                          ('current_rgb_hdf5_sha256','bad')]:
            a=anchor();a[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):gate(a=a)

    def test_h50_short_stream_gap_overlap_and_test_split_refused(self):
        with self.assertRaisesRegex(ValueError,'fewer than 50'):
            gate(rows(49))
        data=rows();data[20]['before_physics_step']+=1
        with self.assertRaisesRegex(ValueError,'interval'):
            gate(data)
        data=rows();data[20]['before_physics_step']-=1
        with self.assertRaisesRegex(ValueError,'interval'):
            gate(data)
        with self.assertRaisesRegex(ValueError,'unapproved split'):
            gate(fold='test')


if __name__=='__main__':unittest.main()

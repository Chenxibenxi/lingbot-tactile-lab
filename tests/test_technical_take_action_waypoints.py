"""Replay proposal != original policy high-level action or approved training labels."""
import unittest
from technical_take_action_waypoints import STRIDE,propose_waypoints,sanitized_card


def rows(n=102):
    result=[]
    for step in range(1,n+1):
        values=[step/1000]*14;values[6]=0.5;values[13]=0.5
        result.append({'physics_step':step,'motion_phase':'expert_dense',
                       'action_index':1 if step<=50 else 2,
                       'drive_target_state_14':values})
    return result


def proposal(data=None,**kwargs):
    return propose_waypoints(rows() if data is None else data,raw_sha256='a'*64,
                             command_sha256='b'*64,**kwargs)


class TechnicalWaypointsTests(unittest.TestCase):
    def test_fixed_stride_proposal_keeps_last_tick_but_proves_zero_take_actions(self):
        p=proposal()
        self.assertEqual([x['source_physics_step'] for x in p['private_waypoints']],
                         [25,50,75,100,102])
        self.assertEqual(p['fixed_proposal_stride_ticks'],STRIDE)
        self.assertEqual(p['actual_take_action_calls'],0)
        self.assertFalse(p['training_approval'])
        self.assertFalse(p['known_seed_replay_is_independent_seed'])
        card=sanitized_card(p)
        self.assertNotIn('private_waypoints',card)
        self.assertEqual(card['proposed_waypoint_count'],5)

    def test_invalid_stride_or_missing_source_provenance_refuses(self):
        with self.assertRaisesRegex(ValueError,'fixed stride'):
            proposal(stride=5)
        with self.assertRaisesRegex(ValueError,'verified technical'):
            propose_waypoints(rows(),raw_sha256='not a sha',command_sha256='b'*64)

    def test_gaps_wrong_action_phase_and_out_of_range_gripper_refuse(self):
        for mutate in (lambda x:x[5].update(physics_step=100),
                       lambda x:x[60].update(action_index=0),
                       lambda x:x[0].update(motion_phase='actual_take_action'),
                       lambda x:x[3]['drive_target_state_14'].__setitem__(6,1.5)):
            data=rows();mutate(data)
            with self.subTest(mutation=mutate),self.assertRaisesRegex(ValueError,'noncontiguous'):
                proposal(data)


if __name__=='__main__':unittest.main()

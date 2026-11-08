import copy
import unittest
from recovery_pair_state_v2 import compare_states, REQUIRED_DIMENSIONS


class StateTest(unittest.TestCase):
    def setUp(self):
        self.state = dict(schema='recovery_observable_state_v2', physics_step=20,
                          values={k:[0.]*n for k,n in REQUIRED_DIMENSIONS.items()},
                          training_approval=False)
        for side in ('left','right'):
            self.state['values'].update({side+'.qpos':[0.,1.], side+'.qvel':[0.,0.],
                side+'.joint.0.test.drive_position':[0.], side+'.joint.0.test.drive_velocity':[0.]})

    def test_exact_and_tiny_drift(self):
        same = compare_states(self.state, copy.deepcopy(self.state))
        self.assertTrue(same['observable_state_equal'])
        self.assertFalse(same['training_approval'])
        changed = copy.deepcopy(self.state)
        changed['values']['bottle1.linear_velocity'][0] = 1e-12
        self.assertFalse(compare_states(self.state, changed)['observable_state_equal'])

    def test_clock_drift(self):
        changed = copy.deepcopy(self.state); changed['physics_step'] += 1
        self.assertFalse(compare_states(self.state, changed)['observable_state_equal'])

    def test_missing_nonfinite_dimension_and_approval(self):
        variants = []
        x = copy.deepcopy(self.state); del x['values']['left.qpos']; variants.append(x)
        x = copy.deepcopy(self.state); x['values']['left.qpos'] = [float('nan'),1.]; variants.append(x)
        x = copy.deepcopy(self.state); x['values']['left.qpos'] = [0.]; variants.append(x)
        x = copy.deepcopy(self.state); x['training_approval'] = True; variants.append(x)
        for x in variants:
            with self.subTest(x=x), self.assertRaises(ValueError):
                compare_states(self.state, x)

    def test_identically_incomplete_states_rejected(self):
        del self.state['values']['bottle2.angular_velocity']
        with self.assertRaises(ValueError):
            compare_states(self.state, self.state)

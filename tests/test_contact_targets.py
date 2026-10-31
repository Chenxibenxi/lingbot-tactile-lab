"""CPU weak labels exist only AFTER real executed-action time intervals."""
import math
import unittest
from contact_targets import executed_interval_targets


def rows():
    return [{'physics_step':step,
             'left':{'points':int(step==2),'load_proxy_n':2. if step==2 else 0.},
             'right':{'points':int(step==5),'load_proxy_n':3. if step==5 else 0.}}
            for step in range(1,7)]


class FutureContactTargetTests(unittest.TestCase):
    def test_exact_executed_action_intervals_no_future_input(self):
        labels=executed_interval_targets(rows(),[(0,0,0,3),(0,1,3,6)],dt_s=.004)
        self.assertEqual(len(labels),2)
        self.assertTrue(labels[0]['left_contact'])
        self.assertFalse(labels[0]['right_contact'])
        self.assertEqual(labels[0]['left_first_step_in_interval'],2)
        self.assertFalse(labels[1]['left_contact'])
        self.assertTrue(labels[1]['right_contact'])
        self.assertEqual(labels[1]['right_first_step_in_interval'],5)
        self.assertEqual(labels[1]['duration_s'],.012)
        self.assertEqual(labels[1]['source'],'future_physics_after_executed_action_not_model_input')
        self.assertEqual(labels[1]['physical_samples'],3)

    def test_missing_or_overlapping_physics_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'missing physics'):
            executed_interval_targets([r for r in rows() if r['physics_step']!=4],[(0,0,0,6)],dt_s=.004)
        with self.assertRaisesRegex(ValueError,'overlapping'):
            executed_interval_targets(rows(),[(0,0,0,3),(0,1,2,6)],dt_s=.004)

    def test_nonfinite_or_negative_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError,'positive finite'):
            executed_interval_targets(rows(),[(0,0,0,3)],dt_s=math.nan)
        bad=rows();bad[1]['left']['load_proxy_n']=math.nan
        with self.assertRaisesRegex(ValueError,'invalid per-side'):
            executed_interval_targets(bad,[(0,0,0,3)],dt_s=.004)
        bad=rows();bad[1]['left']['points']=-1
        with self.assertRaisesRegex(ValueError,'invalid per-side'):
            executed_interval_targets(bad,[(0,0,0,3)],dt_s=.004)


if __name__=='__main__':unittest.main()

import copy,unittest
from compare_feedback_execution_paths import compare_actions
class CompareTest(unittest.TestCase):
 def test_identity_length_and_localized_difference(self):
  a=[dict(command=[i]*14,physics_before=i*2,physics_after=i*2+2,after={s:dict(arm_actual_qpos=[i]*6,arm_drive_targets=[i]*6) for s in ('left','right')}) for i in range(3)]
  self.assertTrue(compare_actions(a,a)['exact_full_execution_match'])
  self.assertFalse(compare_actions(a,a[:2])['exact_full_execution_match'])
  b=copy.deepcopy(a);b[1]['after']['right']['arm_actual_qpos'][0]+=.01
  r=compare_actions(a,b)
  self.assertEqual(r['first_difference_zero_based']['actual_qpos'],1)
  self.assertIsNone(r['first_difference_zero_based']['command'])
if __name__=='__main__':unittest.main()

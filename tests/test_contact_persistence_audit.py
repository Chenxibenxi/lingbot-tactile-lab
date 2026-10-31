"""Count sustained physical contact, not just any contact-positive frame."""
import unittest
from contact_persistence_audit import touch_runs


def rows(points):
    return [{'physics_step':i,'left':{'points':v},'right':{'points':0}}
            for i,v in enumerate(points,1)]


class ContactPersistenceTests(unittest.TestCase):
    def test_sustained_then_interrupted_grasps(self):
        stable=touch_runs(rows([0,1,1,1,1]),'left')
        missed=touch_runs(rows([0,1,1,0,1,0]),'left')
        self.assertEqual((stable['total_contact_steps'],stable['runs'],stable['longest_contiguous_run']), (4,1,4))
        self.assertTrue(stable['final_physics_step_in_contact'])
        self.assertEqual((missed['total_contact_steps'],missed['runs'],missed['longest_contiguous_run']), (3,2,2))
        self.assertFalse(missed['final_physics_step_in_contact'])
        self.assertEqual(touch_runs(rows([0,0]),'right')['runs'],0)

    def test_missing_tick_or_invalid_points_refused(self):
        broken=rows([0,1]);broken[1]['physics_step']=3
        with self.assertRaisesRegex(ValueError,'gap'):
            touch_runs(broken,'left')
        bad=rows([0,-1])
        with self.assertRaisesRegex(ValueError,'invalid contact'):
            touch_runs(bad,'left')


if __name__=='__main__':unittest.main()

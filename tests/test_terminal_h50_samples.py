import unittest
import numpy as np
from terminal_h50_samples import TerminalEpisode
class RealTerminalSource(unittest.TestCase):
    def test_terminal_target_is_real_short_prefix(self):
        ep=TerminalEpisode(100023);s=ep.sample(len(ep)-1,False)
        self.assertEqual(s['target']['real_steps'],1)
        self.assertEqual(int(s['target']['valid_steps'].sum()),1)
        np.testing.assert_allclose(s['target']['actions_14'][0],ep.reader.actions[-1]['command_state_14'],rtol=0,atol=1e-6)
        self.assertNotIn('actions_14',s['input'])
        self.assertTrue(s['metadata']['contact'])
    def test_test_or_validation_relabel_not_allowed(self):
        for seed in (100009,100012,100028,100029):
            with self.assertRaises(ValueError):TerminalEpisode(seed)
        self.assertEqual(TerminalEpisode(100026).fold,'validation')
    def test_no_future_contact_in_terminal_input(self):
        ep=TerminalEpisode(100023)
        for i in (0,len(ep)-1):
            s=ep.sample(i,False);frame=ep.reader.frames[s['metadata']['frame_index']]
            w=ep.reader.windows[frame['obs_index']]
            self.assertLessEqual(w['quality']['last_source_step'],s['input']['capture_physics_step'])
if __name__=='__main__':unittest.main()

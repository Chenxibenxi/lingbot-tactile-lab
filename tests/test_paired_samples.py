import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paired_samples import PairedEpisode


def record(frame, step, future_step, valid):
    return {"task":"lift_pot","seed":0,"episode":0,
            "input":{"frame_index":frame,"capture_physics_step":step,
                     "capture_sim_time_s":step*.004,
                     "command_state_14":[.1]*14,
                     "tactile_history":[[[.1,.2,1.0] for _ in range(4)] for _ in range(5)],
                     "tactile_mask":[[1.0]*4 for _ in range(5)],
                     "tactile_window_valid":valid,
                     "tactile_quality":{"coverage":1.0}},
            "target":{"action_command_14":[.2]*14,
                      "action_source_frame_index":frame+1,
                      "next_frame_physics_step":future_step}}


class PairedSamplesTest(unittest.TestCase):
    def test_only_causal_nonduplicate_frames_and_no_future_input(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td); trial=repo/'data/paired/trial0';(trial/'force').mkdir(parents=True)
            (trial/'force/test.jsonl').touch()
            card={"episode":0,"saved_frames":4,"source_machine_role":"relay_single_rtx3090"}
            examples=[record(0,0,10,False),record(1,10,10,True),record(2,10,25,True)]
            with patch('paired_samples.build_card',return_value=card),patch('paired_samples.build',return_value=iter(examples)):
                ep=PairedEpisode(repo,'trial0',instruction='lift the pot')
                self.assertEqual(len(ep),1)
                sample=ep.sample(0,include_images=False)
            self.assertEqual(sample['metadata']['frame_index'],2)
            self.assertEqual(sample['metadata']['next_frame_step_delta'],15)
            self.assertEqual(sample['input']['tactile_history'].shape,(5,4,3))
            self.assertEqual(sample['input']['tactile_mask'].shape,(5,4))
            self.assertEqual(sample['target']['action_command_14'].shape,(14,))
            self.assertNotIn('action_command_14',sample['input'])
            self.assertNotIn('images_rgb',sample['input'])

    def test_requires_nonempty_instruction(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);trial=repo/'data/paired/trial0';trial.mkdir(parents=True)
            with patch('paired_samples.build_card',return_value={"episode":0}):
                with self.assertRaisesRegex(ValueError,'instruction'):
                    PairedEpisode(repo,'trial0',instruction='  ')


if __name__=='__main__': unittest.main()

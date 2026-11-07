"""Fake controller proof; no RoboTwin/SAPIEN or GPU execution."""
import json
from pathlib import Path
import tempfile
import unittest
from technical_take_action_replay import CAP_8000_TRIAL,approved_physics_cap,run_controller
from technical_take_action_waypoints import SOURCE_ROLE


class FakeProbe:
    step=0
    failed=False


class FakeController:
    def __init__(self,stop_at=3,advance=86):
        self._force_probe=FakeProbe();self._force_probe.step=0
        self.eval_success=False;self.called=0;self.stop_at=stop_at;self.advance=advance
    def take_action(self,command,action_type='qpos'):
        assert action_type=='qpos' and len(command)==14
        self.called+=1;self._force_probe.step+=self.advance
        self.eval_success=self.called==self.stop_at


def proposal():
    return {'source_role':SOURCE_ROLE,'status':'not_executed_not_action_labels_not_training_approved',
            'training_approval':False,'actual_take_action_calls':0,'source_raw_sha256':'a'*64,
            'source_command_sha256':'b'*64,
            'private_waypoints':[{'source_physics_step':(i+1)*25,
                                  'proposed_qpos_command_14':[0.5]*14} for i in range(64)]}


class TechnicalReplayExecutionTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'data/technical_replay'
        parent.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(temporary.cleanup)
        self.output=Path(temporary.name)/'calls.jsonl'

    def test_real_controller_calls_record_new_run_intervals_but_no_training(self):
        task=FakeController(stop_at=3)
        report=run_controller(task,proposal(),output_path=self.output,
                              capture=lambda t:{'capture_physics_step':t._force_probe.step,
                                                'frame_index':t.called})
        self.assertEqual((report['actual_take_action_calls'],report['physics_steps']),(3,258))
        self.assertEqual(report['stopped_reason'],'early_goal_success')
        self.assertEqual(report['eligible_h50_windows_verified'],0)
        self.assertFalse(report['training_approval'])
        records=[json.loads(x) for x in self.output.read_text().splitlines()]
        self.assertEqual(len(records),4)
        self.assertEqual([(x['before_physics_step'],x['after_physics_step']) for x in records[1:]],
                         [(0,86),(86,172),(172,258)])
        with self.assertRaisesRegex(ValueError,'NEW|new technical'):
            run_controller(task,proposal(),output_path=self.output,
                           capture=lambda t:{'capture_physics_step':t._force_probe.step})

    def test_wrong_proposal_source_or_nonadvancing_controller_refuses(self):
        bad=proposal();bad['source_role']='expert_training'
        with self.assertRaisesRegex(ValueError,'source-validated'):
            run_controller(FakeController(),bad,output_path=self.output,
                           capture=lambda t:{'capture_physics_step':t._force_probe.step})
        with self.assertRaisesRegex(RuntimeError,'did not advance'):
            run_controller(FakeController(advance=0),proposal(),output_path=self.output,
                           capture=lambda t:{'capture_physics_step':t._force_probe.step})

    def test_extended_cap_only_valid_for_exact_user_approved_v4_trial(self):
        new=Path(__file__).resolve().parents[1]/'data/technical_replay'/CAP_8000_TRIAL/'calls.jsonl'
        self.assertTrue(approved_physics_cap(self.output,6000))
        self.assertTrue(approved_physics_cap(new,8000))
        self.assertFalse(approved_physics_cap(self.output,8000))
        self.assertFalse(approved_physics_cap(new,8001))
        self.assertFalse(approved_physics_cap(new,True))
        with self.assertRaisesRegex(ValueError,'source-validated'):
            run_controller(FakeController(),proposal(),output_path=self.output,
                           capture=lambda t:{'capture_physics_step':t._force_probe.step},physics_cap=8000)
        self.assertFalse(self.output.exists())

    def test_early_physics_cap_and_wrong_camera_marker_never_claim_success(self):
        task=FakeController(stop_at=100,advance=1000)
        report=run_controller(task,proposal(),output_path=self.output,
                              capture=lambda t:{'capture_physics_step':t._force_probe.step})
        self.assertEqual(report['actual_take_action_calls'],6)
        self.assertEqual(report['stopped_reason'],'bounded_physics_cap')
        self.assertFalse(report['goal_success_reported'])
        with self.assertRaisesRegex(ValueError,'new-run camera'):
            run_controller(FakeController(),proposal(),output_path=self.output.with_name('wrong.jsonl'),
                           capture=lambda t:{'capture_physics_step':t._force_probe.step+1})


if __name__=='__main__':unittest.main()

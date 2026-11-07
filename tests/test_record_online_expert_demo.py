"""No-GPU one-seed writer preflight; never creates a real trial in tests."""
import os
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from record_online_expert_demo import REPO,TRIAL,OVERLAY,verify_fresh_source,run_once,audit_same_run_online_clock


class OnlineExpertRecorderPreflightTests(unittest.TestCase):
    def test_exact_fresh_seed_and_overlay_dry_run_creates_no_artifact(self):
        out=REPO/'data/paired'/TRIAL
        if out.exists():
            with self.assertRaisesRegex(ValueError,'only one fresh'):
                verify_fresh_source(out)
        else:
            result=verify_fresh_source(out)
            self.assertEqual((result['seed'],result['new_trial_id']),(100022,TRIAL))
            self.assertFalse(result['gpu_run_allowed_now'])
            self.assertFalse(out.exists())
        with self.assertRaisesRegex(ValueError,'only one fresh'):
            verify_fresh_source(REPO/'data/technical_replay'/TRIAL)
        with self.assertRaisesRegex(ValueError,'only one fresh'):
            verify_fresh_source(REPO/'data/paired/relay_pickdiverse_s100021_online_h50_expert0')

    def test_direct_run_without_approved_flags_or_own_lease_never_creates_trial(self):
        out=REPO/'data/paired'/TRIAL
        if out.exists():self.skipTest('unique GPU pilot was already executed')
        with patch.dict(os.environ,{'TACTILE_ONLINE_EXPERT_PILOT_APPROVED':'',
                                    'TACTILE_POLICY_EQUIVALENT_EXPERT':''}):
            with self.assertRaisesRegex(RuntimeError,'explicit one-seed'):
                run_once(out,OVERLAY)
        self.assertFalse(out.exists())
        env={'TACTILE_ONLINE_EXPERT_PILOT_APPROVED':'1',
             'TACTILE_POLICY_EQUIVALENT_EXPERT':'1',
             'TACTILE_SOURCE_MACHINE_ROLE':'relay_single_rtx3090',
             'FORCE_PROBE_EXPERT_DIR':str(out/'force')}
        with patch.dict(os.environ,env),patch('record_online_expert_demo.Path.cwd',return_value=OVERLAY), \
             patch('record_online_expert_demo.verify_gpu_lease',side_effect=RuntimeError('no own GPU lease')):
            with self.assertRaisesRegex(RuntimeError,'no own GPU lease'):
                run_once(out,OVERLAY)
        self.assertFalse(out.exists())

    def test_action_camera_physics_sidecar_links_at_fifty_real_intervals(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=root/'trial.jsonl'
            rows=[{'physics_step':i} for i in range(51)]
            frames=[{'schema':'frames'}]+[{'frame_index':i,'physics_step':i,
                                           'obs_index':i} for i in range(50)]
            obs=[{'schema':'obs'}]+[{'physics_step':i} for i in range(50)]
            acts=[{'action_index':i,'camera_frame_index':i,'capture_physics_step':i,
                   'before_physics_step':i,'after_physics_step':i+1,
                   'controller_method':'Base_Task.take_action',
                   'observed_success':i==49,'training_approval':False} for i in range(50)]
            def save():
                raw.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
                raw.with_suffix('.frames.jsonl').write_text('\n'.join(json.dumps(x) for x in frames)+'\n')
                raw.with_suffix('.obs.jsonl').write_text('\n'.join(json.dumps(x) for x in obs)+'\n')
            save()
            result=audit_same_run_online_clock(raw,acts,50)
            self.assertEqual(result['real_highlevel_call_count'],50)
            self.assertFalse(result['training_approval'])
            frames[9]['frame_index']=4;save()
            with self.assertRaisesRegex(ValueError,'mismatch'):
                audit_same_run_online_clock(raw,acts,50)
            frames[9]['frame_index']=8;acts[9]['before_physics_step']=11;save()
            with self.assertRaisesRegex(ValueError,'mismatch'):
                audit_same_run_online_clock(raw,acts,50)

    def _exercise_fifty_call_collector(self,fail_final=False):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'fresh';raw=out/'force/fake.jsonl'
            rows=[{'action_index':i,'phase':'grasp','camera_frame_index':i,
                   'capture_physics_step':i,'before_physics_step':i,'after_physics_step':i+1,
                   'controller_method':'Base_Task.take_action','command_state_14':[0.0]*14,
                   'joint_tracking_max_abs_difference':0.0,'observed_success':i==49,
                   'training_approval':False} for i in range(50)]
            class FakeTask:
                plan_success=True;eval_success=True;closed=False
                def setup_demo(self,**unused):
                    raw.parent.mkdir(parents=True)
                    raw.write_text('\n'.join(json.dumps({'physics_step':i}) for i in range(51))+'\n')
                    raw.with_suffix('.frames.jsonl').write_text('\n'.join(json.dumps(x) for x in
                        [{'schema':'frames'}]+[{'frame_index':i,'obs_index':i,'physics_step':i}
                                              for i in range(50)])+'\n')
                    raw.with_suffix('.obs.jsonl').write_text('\n'.join(json.dumps(x) for x in
                        [{'schema':'obs'}]+[{'physics_step':i} for i in range(50)])+'\n')
                    self._force_probe=SimpleNamespace(raw_path=raw,saved_frames=50,step=0,failed=False)
                def check_success(self):return True
                def close_env(self):self.closed=True
                def merge_pkl_to_hdf5_video(self):
                    assert self.closed
                    dest=out/'demonstration/data/episode0.hdf5';dest.parent.mkdir(parents=True)
                    dest.write_bytes(b'synthetic CPU fixture, never an expert HDF5')
            fake=FakeTask()
            result={'schema':'fake_controller','actual_take_action_calls':50,'physics_steps':50,
                    'goal_success_reported':True,'stopped_reason':'early_goal_success',
                    'training_approval':False,'private_action_rows_in_memory_only':rows}
            validated={'task':'pick_diverse_bottles','seed':100022,'episode':0,
                       'physics_steps':50,'saved_frames':50,'sha256':{'test':'fixture'}}
            calls=[]
            def execute_mock(robot,capture,*,physics_cap):
                assert physics_cap==10000
                robot._force_probe.step=50
                return result
            def validate_mock(*args):
                calls.append(args)
                if fail_final and len(calls)==2:raise ValueError('bad final completion')
                return validated
            manifest={'original_base_task_sha256':'a'*64,
                      'original_bottle_task_sha256':'b'*64,
                      'original_robot_planner_sha256':'c'*64}
            flags={'TACTILE_ONLINE_EXPERT_PILOT_APPROVED':'1',
                   'TACTILE_POLICY_EQUIVALENT_EXPERT':'1',
                   'TACTILE_SOURCE_MACHINE_ROLE':'relay_single_rtx3090',
                   'FORCE_PROBE_EXPERT_DIR':str(out/'force')}
            with patch.dict(os.environ,flags),patch('record_online_expert_demo.Path.cwd',return_value=OVERLAY), \
                 patch('record_online_expert_demo.verify_fresh_source',return_value=manifest), \
                 patch('record_online_expert_demo.verify_gpu_lease',return_value=True), \
                 patch('record_online_expert_demo._config',return_value={}), \
                 patch('importlib.import_module',return_value=SimpleNamespace(pick_diverse_bottles=lambda:fake)), \
                 patch('record_online_expert_demo.execute_online_expert',side_effect=execute_mock), \
                 patch('record_online_expert_demo.validate',side_effect=validate_mock):
                if fail_final:
                    with self.assertRaisesRegex(ValueError,'bad final completion'):
                        run_once(out,OVERLAY)
                else:
                    with redirect_stdout(StringIO()):
                        self.assertEqual(run_once(out,OVERLAY),0)
            self.assertEqual(len(calls),2)
            self.assertEqual((out/'completion.json').is_file(),not fail_final)
            self.assertEqual((out/'failure.json').is_file(),fail_final)
            self.assertTrue((out/'online_actions.jsonl').is_file())

    def test_synthetic_cpu_fifty_call_success_creates_only_candidate_completion(self):
        self._exercise_fifty_call_collector()

    def test_synthetic_cpu_final_validation_failure_never_writes_completion(self):
        self._exercise_fifty_call_collector(fail_final=True)


if __name__=='__main__':unittest.main()

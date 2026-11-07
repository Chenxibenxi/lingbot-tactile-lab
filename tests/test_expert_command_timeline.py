"""CPU-only per-physics-step expert DRIVE TARGET sidecar; no GPU/HDF5/raw bytes."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from expert_command_timeline import ExpertCommandTimeline, validate_commands, ORIGIN
from validate_expert_data import validate, sha256


class FakeRobot:
    def __init__(self):self.value=0.0
    def get_left_arm_jointState(self):return [self.value]*6+[0.5]
    def get_right_arm_jointState(self):return [self.value+1]*6+[0.25]


class CommandTimelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'force').mkdir();self.raw=self.root/'force/lift_pot_123_0_unique.jsonl'
        self.frames=self.raw.with_suffix('.frames.jsonl')
        self.commands=self.root/'commands'/f'{self.raw.stem}.commands.jsonl'
        self.header={'schema':'phizju_force_probe_v1','task':'lift_pot','seed':123,'episode':0,
                     'dt_s':.004,'clock_origin':ORIGIN}
        self.motion={'action_index':1,'motion_index':1,'motion_phase':'expert_dense'}
        self.robot=FakeRobot()
        self.write_fixture()

    def dump(self,path,rows):path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    def rows(self,path):return [json.loads(x) for x in path.read_text().splitlines()]

    def write_fixture(self):
        raw=[self.header]
        for step in range(1,4):raw.append({'physics_step':step,**self.motion})
        self.dump(self.raw,raw)
        writer=ExpertCommandTimeline(self.raw,self.root/'commands',task='lift_pot',seed=123,
                                     episode=0,dt_s=.004,clock_origin=ORIGIN)
        states=[]
        for step in range(1,4):
            self.robot.value=float(step)
            writer.write_after_step(step,self.robot,**self.motion)
            states.append(self.robot.get_left_arm_jointState()+self.robot.get_right_arm_jointState())
        writer.close()
        self.dump(self.frames,[{'schema':'phizju_saved_frames_v1','probe_file':self.raw.name,
                                'task':'lift_pot','seed':123,'episode':0,'dt_s':.004,
                                'clock_origin':ORIGIN},
                               {'frame_index':0,'physics_step':0,'command_state_14':[0.]*14},
                               {'frame_index':1,'physics_step':2,'command_state_14':states[1]},
                               {'frame_index':2,'physics_step':3,'command_state_14':states[2]}])

    def test_matches_per_physics_step_and_marks_unsampled_initial_frame(self):
        result=validate_commands(self.raw,self.frames,self.commands)
        self.assertEqual(result['status'],'source_checked_not_training_approved')
        self.assertEqual(result['physics_steps'],3)
        self.assertEqual(result['initial_frames_without_instrumented_command'],1)
        self.assertEqual(self.rows(self.commands)[2]['physics_step'],2)
        self.assertEqual(self.rows(self.commands)[2]['drive_target_state_14'][6],.5)
        self.assertNotIn('approved_for_training',str(result))

    def test_missing_duplicate_or_changed_command_fails_closed(self):
        rows=self.rows(self.commands)
        self.dump(self.commands,[rows[0],rows[1],rows[3]])
        with self.assertRaisesRegex(ValueError,'count'):
            validate_commands(self.raw,self.frames,self.commands)
        rows[2]['physics_step']=1;self.dump(self.commands,rows)
        with self.assertRaisesRegex(ValueError,'mismatch at step 2'):
            validate_commands(self.raw,self.frames,self.commands)
        rows[2]['physics_step']=2;rows[2]['drive_target_state_14'][0]+=0.1
        self.dump(self.commands,rows)
        with self.assertRaisesRegex(ValueError,'same-step command'):
            validate_commands(self.raw,self.frames,self.commands)

    def test_clock_source_or_gripper_tamper_rejected(self):
        rows=self.rows(self.commands);rows[0]['probe_file']='another.jsonl';self.dump(self.commands,rows)
        with self.assertRaisesRegex(ValueError,'header'):
            validate_commands(self.raw,self.frames,self.commands)
        rows[0]['probe_file']=self.raw.name;rows[1]['drive_target_state_14'][13]=2.
        self.dump(self.commands,rows)
        with self.assertRaisesRegex(ValueError,'mismatch at step 1'):
            validate_commands(self.raw,self.frames,self.commands)
        rows[1]['drive_target_state_14'][13]=.25;rows[1]['drive_target_state_14'][0]=float('nan')
        self.commands.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaisesRegex(ValueError,'mismatch at step 1'):
            validate_commands(self.raw,self.frames,self.commands)

    def test_path_isolated_and_exclusive_creation(self):
        with self.assertRaisesRegex(ValueError,'sibling'):
            ExpertCommandTimeline(self.raw,self.root/'wrong',task='lift_pot',seed=123,
                                  episode=0,dt_s=.004,clock_origin=ORIGIN)
        with self.assertRaises(FileExistsError):
            ExpertCommandTimeline(self.raw,self.root/'commands',task='lift_pot',seed=123,
                                  episode=0,dt_s=.004,clock_origin=ORIGIN)
        bad=self.root/'elsewhere/escape.commands.jsonl';bad.parent.mkdir()
        with self.assertRaisesRegex(ValueError,'sibling'):
            validate_commands(self.raw,self.frames,bad)

    def test_writer_rejects_skipped_tick_invalid_motion_or_action(self):
        assert self.commands.resolve() == (self.root.resolve()/'commands'/f'{self.raw.stem}.commands.jsonl')
        self.commands.unlink()
        writer=ExpertCommandTimeline(self.raw,self.root/'commands',task='lift_pot',seed=123,
                                     episode=0,dt_s=.004,clock_origin=ORIGIN)
        try:
            with self.assertRaisesRegex(ValueError,'noncontiguous'):
                writer.write_after_step(2,self.robot,**self.motion)
            with self.assertRaisesRegex(ValueError,'expert physics'):
                writer.write_after_step(1,self.robot,action_index=1,motion_index=1,motion_phase='policy_action')
            self.robot.value=float('nan')
            with self.assertRaisesRegex(ValueError,'14D'):
                writer.write_after_step(1,self.robot,**self.motion)
            self.robot.value=1.
            writer.write_after_step(1,self.robot,**self.motion)
        finally:writer.close()

    def test_completion_binds_optional_command_sha_and_revalidates_full_source(self):
        import numpy as np
        import h5py
        raw=self.rows(self.raw)
        raw[0]['side_links']={'left':['fl7'],'right':['fr7']}
        for i,row in enumerate(raw[1:],start=1):
            row['sim_time_s']=round(i*.004,8)
            row['left']={'fingers':{'fl7':{}}};row['right']={'fingers':{'fr7':{}}}
        self.dump(self.raw,raw)
        obs=self.raw.with_suffix('.obs.jsonl')
        self.dump(obs,[{'schema':'phizju_obs_marker_v1','probe_file':self.raw.name,
                        'task':'lift_pot','seed':123,'episode':0,'dt_s':.004},
                       *({'obs_index':i,'physics_step':step} for i,step in enumerate((0,2,3)))])
        frames=self.rows(self.frames)
        for i,row in enumerate(frames[1:]):
            row['obs_index']=i
            row['actual_joints']={side:{'names':[side+'_joint'],'qpos':[.1],'qvel':[0.]}
                                  for side in ('left','right')}
        self.dump(self.frames,frames)
        hdf=self.root/'demonstration/data/episode0.hdf5';hdf.parent.mkdir(parents=True)
        with h5py.File(hdf,'w') as out:
            out.create_dataset('joint_action/vector',data=np.asarray([r['command_state_14'] for r in frames[1:]]))
            for cam in ('head_camera','left_camera','right_camera'):
                out.create_dataset(f'observation/{cam}/rgb',data=np.asarray([0,1,2]))
        result=validate(self.raw,obs,self.frames,hdf)
        gate=self.root/'expert_success_gate.json';gate.write_text(json.dumps({
            'schema':'phizju_expert_success_gate_v1','success':True,'planner_ok':True,'goal_ok':True,
            'source_machine_role':'relay_single_rtx3090','source_hostname':'relay-test',
            'task':'lift_pot','seed':123,'episode':0,'saved_frame_markers':3,
            'source_sha256':{key:result['sha256'][key] for key in ('raw','obs','frames')},
            'command_timeline_sha256':sha256(self.commands)}))
        audit=validate_commands(self.raw,self.frames,self.commands)
        completion=self.root/'completion.json';completion.write_text(json.dumps({
            'schema':'phizju_expert_episode_complete_v1','success':True,'task':'lift_pot',
            'seed':123,'episode':0,'source_machine_role':'relay_single_rtx3090',
            'source_hostname':'relay-test','success_gate_sha256':sha256(gate),
            'sha256':result['sha256'],
            'command_timeline':{'schema':audit['schema'],'status':audit['status'],
                'sha256':audit['source_sha256']['commands'],
                'physics_steps':audit['physics_steps'],
                'initial_frames_without_instrumented_command':1}}))
        self.assertEqual(validate(self.raw,obs,self.frames,hdf,completion)['saved_frames'],3)
        commands=self.rows(self.commands);commands[2]['drive_target_state_14'][0]+=1
        self.dump(self.commands,commands)
        with self.assertRaisesRegex(ValueError,'same-step command'):
            validate(self.raw,obs,self.frames,hdf,completion)


if __name__=='__main__':unittest.main()

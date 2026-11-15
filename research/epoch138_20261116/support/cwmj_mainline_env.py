"""Opt-in environment bridge for the restricted A3 development candidate."""
import copy
import numpy as np
from cwmj_mainline_execution import FrozenA3Kernel,apply_b1_contact
from cwmj_m2_observation_env import MAX_COMMAND_HISTORY,MAX_TACTILE_INTERVALS

class MainlineFrozenEnv:
    def __init__(self,env,inputs,reference):
        self.owns_env=True;self.env=env;self.inputs=copy.deepcopy({'physical':inputs['physical']});self.reference=copy.deepcopy(reference)
        self.binding=apply_b1_contact(env.model);self.kernel=FrozenA3Kernel(env,self.inputs,self.reference)
    def reset(self,seed=0):
        observation=self.env.reset(seed=seed)
        self.kernel=FrozenA3Kernel(self.env,self.inputs,self.reference)
        return observation
    def observe(self):return self.env.observe()
    def close(self):
        if self.owns_env:return self.env.close()
    def clone(self):
        result=copy.copy(self);result.owns_env=False;result.env=copy.copy(self.env);result.kernel=self.kernel.clone()
        result.env.data=result.kernel.env.data
        # These are all interval/command/diagnostic histories changed by step/reset.
        for name in ['command_history','last_force_interval','_last_interval_substeps','_tactile_intervals','_diagnostic_labels','_diagnostic_success_truth']:
            if hasattr(self.env,name):setattr(result.env,name,copy.deepcopy(getattr(self.env,name)))
        result.kernel.env=result.env
        # A preflight clone is physics-only and must never close shared renderer resources.
        result.env.render_enabled=False
        return result
    def step(self,action,*,gripper_units='physical_m'):
        e=self.env;validated=e.validate_action(action,gripper_units=gripper_units)
        self.kernel.replace_target(validated['action_14d_physical_m'])
        start=float(e.data.time);samples=[];execution=[]
        for i in range(25):
            before=float(e.data.time);row=self.kernel.substep();execution.append(row)
            sample=e._force_sample_substep();sample.update(substep=i,sim_time_start_s=before,sim_time_end_s=float(e.data.time),finger_qpos_m={n:float(e.data.qpos[a]) for n,a in e.finger_joint_qposadr.items()})
            samples.append(sample)
            if row['stop']:break
        end=float(e.data.time);complete=len(samples)==25 and not self.kernel.stopped
        interval=e._summarise_force_interval(samples,start,end)
        if not complete:
            interval.update(sensor_valid=False,contact_present=None,max_normal_N=None,missing_policy='partial or stopped control interval: invalid for policy use')
            for finger in interval['per_finger'].values():finger.update(max_normal_N=None,contact_present=None)
        e.last_force_interval=interval;e._last_interval_substeps=samples;e._tactile_intervals.append(interval);e._tactile_intervals=e._tactile_intervals[-MAX_TACTILE_INTERVALS:]
        e._tactile_invalidated_reason=None if complete else 'execution_stopped'
        effective=e.validate_action(execution[-1]['effective'],gripper_units='physical_m')
        e.command_history.append({'control_tick':e.control_tick,'sim_time_s':start,'action_14d_normalized':effective['action_14d_normalized'],'action_14d_physical_m':effective['action_14d_physical_m'],'gripper_units':'normalized','normalized_derivable':bool(effective['normalized_derivable']),'flags':effective['flags'],'legal':True})
        e.command_history=e.command_history[-MAX_COMMAND_HISTORY:];e.control_tick+=1;e.step_index+=1
        return e.observe(),{'raw_action':validated,'execution_substeps':execution,'force_interval':interval,'sim_time_before_s':start,'sim_time_after_s':end,'complete':complete,'stop':self.kernel.stopped,'previous_command_semantics':'last effective driver target, not raw policy target'}

class MainlineContinuousEnv(MainlineFrozenEnv):
    """Development candidate: continuous targets plus mandatory copied prefix check."""
    def __init__(self,env):
        from cwmj_mainline_execution import ContinuousTargetKernel
        self.owns_env=True;self.env=env;self.binding=apply_b1_contact(env.model)
        self.kernel=ContinuousTargetKernel(env)
    def reset(self,seed=0):
        from cwmj_mainline_execution import ContinuousTargetKernel
        observation=self.env.reset(seed=seed);self.kernel=ContinuousTargetKernel(self.env);return observation
    def checked_step(self,action,*,gripper_units='physical_m'):
        trial=self.clone()
        predicted,transition=trial.step(action,gripper_units=gripper_units)
        if not transition['complete']:
            return None,{'accepted':False,'preflight':transition,'actual_advanced':False}
        observation,actual=self.step(action,gripper_units=gripper_units)
        equal=actual==transition and np.array_equal(self.env.data.qpos,trial.env.data.qpos) and np.array_equal(self.env.data.qvel,trial.env.data.qvel)
        if not equal:raise RuntimeError('preflight and execution diverged')
        return observation,{'accepted':True,'actual':actual,'preflight_equal':True,'scope':'one 50ms prefix, not full path clearance'}

"""Opt-in process-local, flush-on-event expert journal; no source or action changes."""
from pathlib import Path
import json
import numpy as np

class ExpertJournal:
    def __init__(self,task,api,path):
        self.task=task;self.api=api;self.path=Path(path);self.phase=None;self.action_index=0;self.originals={}
    def event(self,kind,**values):
        probe=getattr(self.task,'_force_probe',None)
        row={'event':kind,'phase':self.phase,'action_index':self.action_index,'physics_step':getattr(probe,'step',None),**values}
        self.file.write(json.dumps(row,allow_nan=False)+'\n');self.file.flush()
    def observation(self):
        out={}
        for side in ('left','right'):
            try:out[side+'_measured_qpos']=[float(v) for v in getattr(self.task.robot,f'get_{side}_arm_real_jointState')()]
            except Exception as exc:out[side+'_read_error']=type(exc).__name__
        for name in ('bottle1','bottle2'):
            try:out[name+'_functional_point']=np.asarray(getattr(self.task,name).get_functional_point(0)).tolist()
            except Exception as exc:out[name+'_read_error']=type(exc).__name__
        return out
    def __enter__(self):
        self.file=self.path.open('x');self.event('journal_start',role='diagnostic_replay_not_expert_training_data')
        self.originals={'paired':self.api._paired_actions,'plan':self.api._plan,'take_action':self.task.take_action}
        def paired(task,stage):
            self.phase=stage;self.event('stage_enter',observation=self.observation())
            return self.originals['paired'](task,stage)
        def plan(robot,action,arm):
            self.event('plan_enter',arm=arm,target_pose=np.asarray(action.target_pose).tolist(),observation=self.observation())
            try:result=self.originals['plan'](robot,action,arm)
            except Exception as exc:
                self.event('plan_exception',arm=arm,type=type(exc).__name__,message=str(exc),observation=self.observation());raise
            self.event('plan_return',arm=arm,path_length=len(result));return result
        def take_action(command,*args,**kwargs):
            self.event('action_enter',command=np.asarray(command).tolist(),observation=self.observation())
            result=self.originals['take_action'](command,*args,**kwargs)
            self.event('action_return',observation=self.observation(),success=bool(getattr(self.task,'eval_success',False)))
            self.action_index+=1;return result
        self.api._paired_actions=paired;self.api._plan=plan;self.task.take_action=take_action
        return self
    def __exit__(self,kind,value,tb):
        self.api._paired_actions=self.originals['paired'];self.api._plan=self.originals['plan'];self.task.take_action=self.originals['take_action']
        try:self.event('journal_end',exception=None if kind is None else kind.__name__)
        finally:self.file.close()
        return False

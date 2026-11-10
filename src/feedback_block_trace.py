"""Fail-closed provenance trace for per-step inference/block-source execution."""
import json
import numpy as np
from feedback_block_intervention import BlockIntervention

class BlockTrace:
    def __init__(self,path,mode,block_size=10):
        self.selector=BlockIntervention(mode,block_size)
        self.f=open(path,'x')
        self.pending=None
        self.step=0
        self.count=0
        self._write(dict(event='schema',schema='feedback_block_trace_v1',mode=mode,
                         block_size=block_size,training_approved=False))
    def _write(self,row):
        self.f.write(json.dumps(row,allow_nan=False)+'\n');self.f.flush()
    def request(self,*,index,physics_step,commands,physical_state):
        if self.pending is not None or index!=self.count or physics_step!=self.step:
            raise ValueError('request/action/clock sequence')
        command,source=self.selector.select(commands,action_index=index)
        self._write(dict(event='prediction',index=index,physics_step=physics_step,
                         commands=np.asarray(commands).tolist(),physical_state=physical_state))
        self.pending=(command,source)
        return command.copy(),dict(source)
    def action(self,*,command,physics_before,physics_after,before,after):
        if self.pending is None:raise ValueError('prediction required')
        expected,source=self.pending
        if not np.array_equal(command,expected):raise ValueError('wrong source command')
        if physics_before!=self.step or physics_after<=physics_before:raise ValueError('physical clock')
        self._write(dict(event='action',command=np.asarray(command).tolist(),source=source,
                         physics_before=physics_before,physics_after=physics_after,before=before,after=after))
        self.pending=None;self.step=physics_after;self.count+=1
    def close(self,*,complete=False):
        if self.f.closed:return
        if complete and (self.pending is not None or not self.count):raise ValueError('incomplete trace')
        self._write(dict(event='terminal',complete=bool(complete)));self.f.close()

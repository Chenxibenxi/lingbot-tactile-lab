"""Separate passive trace of decoded prediction blocks and actual physical states."""
import json
import numpy as np

class PredictionTrace:
    def __init__(self,path):
        self.f=open(path,'x');self.request_index=-1;self.applied=0;self.block=None;self.step=-1
        self._write(dict(event='schema',schema='feedback_prediction_trace_v1',training_approved=False))
    def _write(self,row):
        self.f.write(json.dumps(row,allow_nan=False)+'\n');self.f.flush()
    def request(self,*,index,physics_step,commands,physical_state):
        block=np.array(commands,copy=True)
        if block.shape!=(50,14) or not np.isfinite(block).all():raise ValueError('finite decoded H50 required')
        if index!=self.request_index+1 or physics_step<self.step:raise ValueError('request order/clock mismatch')
        if self.block is not None and self.applied==0:raise ValueError('previous request unexecuted')
        self._write(dict(event='prediction',index=index,physics_step=physics_step,commands=block.tolist(),physical_state=physical_state))
        self.block=block;self.request_index=index;self.applied=0;self.step=physics_step
    def action(self,*,command,physics_before,physics_after,before,after):
        if self.block is None or self.applied>=50:raise ValueError('prediction required')
        if not np.array_equal(np.asarray(command),self.block[self.applied]):raise ValueError('command differs from predicted prefix')
        if physics_before!=self.step or physics_after<=physics_before:raise ValueError('noncontiguous physical clock')
        self._write(dict(event='action',request_index=self.request_index,index_in_request=self.applied,
            physics_before=physics_before,physics_after=physics_after,command=np.asarray(command).tolist(),before=before,after=after))
        self.applied+=1;self.step=physics_after
    def close(self,*,complete=False):
        if self.f.closed:return
        if complete and (self.block is None or self.applied==0):raise ValueError('cannot complete empty trace')
        self._write(dict(event='terminal',complete=bool(complete)));self.f.close()

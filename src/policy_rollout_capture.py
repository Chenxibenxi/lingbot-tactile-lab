"""Opt-in same-episode RGB/tactile/EXECUTED-command capture for policy studies.

This records *self-generated model rollout*, NOT expert correction targets. Merely
having images and actions cannot turn it into an approved supervised dataset.
The original RoboTwin client must call begin_request at the actual infer boundary
and applied_action AFTER each real take_action. No source checkout is patched.
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
from tactile_rpc import encode_tactile_window

CAMERAS={"head":"observation.images.cam_high",
         "left":"observation.images.cam_left_wrist",
         "right":"observation.images.cam_right_wrist"}


class PolicyRolloutWriter:
    def __init__(self, output: Path, *, task: str, seed: int, episode: int):
        self.output=Path(output)
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self._f=h5py.File(self.output,"x")  # never mix trials or overwrite
        self._f.attrs.update(schema="phizju_policy_rollout_capture_v1",task=task,
                             seed=int(seed),episode=int(episode),
                             target_semantics="executed_self_rollout_not_expert_or_training_approved",
                             complete=False)
        self.requests=0
        self.actions=0
        self._last_step=-1
        self._active_request=None
        self._active_actions=0

    def _append(self,name,value,dtype=None,compression=None):
        array=np.asarray(value,dtype=dtype)
        if name not in self._f:
            self._f.create_dataset(name,shape=(0,)+array.shape,
                maxshape=(None,)+array.shape,dtype=array.dtype,
                chunks=(1,)+array.shape,compression=compression)
        ds=self._f[name]
        if ds.shape[1:]!=array.shape:
            raise ValueError(f"changing capture geometry: {name}")
        ds.resize((len(ds)+1,)+ds.shape[1:])
        ds[-1]=array

    def begin_request(self, request:dict, observation:dict, window:dict|None=None):
        """Capture only pixels/state actually supplied to THIS visual policy call."""
        if self._active_request is not None and not self._active_actions:
            raise ValueError("previous request has no applied action")
        index=request.get('request_index')
        step=request.get('physics_step')
        if type(index)!=int or index!=self.requests or type(step)!=int or step<self._last_step:
            raise ValueError("policy request index/physics time not sequential")
        if window is not None and window.get('anchor_physics_step')!=step:
            raise ValueError("tactile window does not belong to this inference step")
        pixels={}
        for name,key in CAMERAS.items():
            array=np.asarray(observation[key])
            if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[2]!=3 or min(array.shape[:2])<=0:
                raise ValueError("policy RGB must be uint8 HWC three-channel")
            pixels[name]=array
        state=np.asarray(observation['observation.state'],dtype=np.float32)
        if state.shape!=(14,) or not np.isfinite(state).all():
            raise ValueError("current command state must be finite 14D")
        payload=encode_tactile_window(window)
        for name,array in pixels.items():
            self._append(f'images/{name}',array,compression='lzf')
        self._append('state_14',state)
        for field,name in (('history','history'),('mask','coverage'),
                           ('age_ms','age_ms'),('valid','window_valid')):
            self._append('tactile/'+name,payload['observation.tactile.'+field])
        self._append('request/obs_index',np.int64(request['obs_index']))
        self._append('request/physics_step',np.int64(step))
        self.requests+=1
        self._last_step=step
        self._active_request=index
        self._active_actions=0
        self._f.flush()

    def applied_action(self, action, *, physics_step_before:int, physics_step_after:int):
        """Only label the exact 14D command passed to an ACTUAL sim action."""
        if self._active_request is None:
            raise ValueError("action without a policy request")
        if (type(physics_step_before)!=int or type(physics_step_after)!=int or
                physics_step_before<self._last_step or physics_step_after<=physics_step_before):
            raise ValueError("applied action must move forward on the physics clock")
        vector=np.asarray(action,dtype=np.float32)
        if vector.shape!=(14,) or not np.isfinite(vector).all():
            raise ValueError("applied command must be finite 14D")
        self._append('applied/command_14',vector)
        for name,value in (('request_index',self._active_request),
                           ('index_in_request',self._active_actions),
                           ('physics_step_before',physics_step_before),
                           ('physics_step_after',physics_step_after)):
            self._append('applied/'+name,np.int64(value))
        self._last_step=physics_step_after
        self.actions+=1
        self._active_actions+=1
        self._f.flush()

    def close(self, *, observed_success:bool|None=None):
        if self._f is None:
            return
        self._f.attrs['policy_reported_success']=(-1 if observed_success is None else int(bool(observed_success)))
        req_paths=[f'images/{side}' for side in CAMERAS]+[
            'state_14','tactile/history','tactile/coverage','tactile/age_ms',
            'tactile/window_valid','request/obs_index','request/physics_step']
        action_paths=['applied/'+name for name in ('command_14','request_index',
            'index_in_request','physics_step_before','physics_step_after')]
        self._f.attrs['complete']=bool(self.requests and self._active_actions and
            all(path in self._f and len(self._f[path])==self.requests for path in req_paths) and
            all(path in self._f and len(self._f[path])==self.actions for path in action_paths))
        self._f.flush()
        self._f.close()
        self._f=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

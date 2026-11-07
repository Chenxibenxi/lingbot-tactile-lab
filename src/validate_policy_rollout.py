"""Validate same-run policy RGB+ACTUALLY-applied actions+causal contact.

This evidence is a self-rollout, NOT expert corrective supervision. No approval
for training can be inferred from a successful outcome or capture completeness.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

from align_temporal import align
from tactile_rpc import encode_tactile_window
from validate_expert_data import read_jsonl,sha256
from validate_policy_trace import validate as validate_trace


def validate(raw,obs,policy,hdf5, *, allow_policy_calls_on_expert_overlay=False):
    raw,obs,policy,hdf5=map(Path,(raw,obs,policy,hdf5))
    trace=validate_trace(raw,obs,policy,allow_policy_calls_on_expert_overlay=allow_policy_calls_on_expert_overlay)
    req=read_jsonl(policy)[1:]
    markers=read_jsonl(obs)[1:]
    byobs={row['obs_index']:row for row in markers}
    windows={w['observation_marker_index']:w for w in align(raw,obs)}
    physics_rows=read_jsonl(raw)
    last_raw_step=physics_rows[-1]['physics_step']
    with h5py.File(hdf5,'r') as f:
        if (f.attrs.get('schema')!='phizju_policy_rollout_capture_v1' or
                'not_expert_or_training_approved' not in f.attrs.get('target_semantics','') or
                not f.attrs.get('complete',False) or
                (f.attrs.get('task'),f.attrs.get('seed'),f.attrs.get('episode')) !=
                (trace['task'],trace['seed'],trace['episode'])):
            raise ValueError('HDF5 is not a complete same-run self-rollout capture')
        n=len(req)
        fields=['images/head','images/left','images/right','state_14','tactile/history',
                'tactile/coverage','tactile/age_ms','tactile/window_valid',
                'request/obs_index','request/physics_step']
        if any(path not in f or len(f[path])!=n for path in fields):
            raise ValueError('HDF5 camera/state/tactile/request count mismatch')
        for i,row in enumerate(req):
            obs_index=row['obs_index'];step=row['physics_step']
            if (f['request/obs_index'][i]!=obs_index or f['request/physics_step'][i]!=step or
                    f['state_14'][i].shape!=(14,) or not np.isfinite(f['state_14'][i]).all()):
                raise ValueError('policy HDF5 request step or state mismatch')
            for side,camera in (('head','head_camera'),('left','left_camera'),('right','right_camera')):
                image=f['images/'+side][i]
                if image.dtype!=np.uint8 or tuple(image.shape)!=tuple(byobs[obs_index]['rgb_shapes'][camera]):
                    raise ValueError('captured RGB geometry not linked to camera marker')
            expected=encode_tactile_window(windows[obs_index])
            for suffix,name in (('history','history'),('mask','coverage'),
                                ('age_ms','age_ms'),('valid','window_valid')):
                if not np.array_equal(f['tactile/'+name][i],
                                      expected['observation.tactile.'+suffix]):
                    raise ValueError('captured tactile differs from same-run causal window')
        action_fields=['applied/command_14','applied/request_index','applied/index_in_request',
                       'applied/physics_step_before','applied/physics_step_after']
        if any(path not in f for path in action_fields):
            raise ValueError('missing actually applied commands')
        count=len(f['applied/command_14'])
        if count<=0 or any(len(f[path])!=count for path in action_fields):
            raise ValueError('incomplete applied-action stream')
        per_request=[0]*n
        last_end=-1
        for i in range(count):
            index=int(f['applied/request_index'][i]); before=int(f['applied/physics_step_before'][i]);after=int(f['applied/physics_step_after'][i])
            vector=f['applied/command_14'][i]
            if (not 0<=index<n or f['applied/index_in_request'][i]!=per_request[index] or
                    vector.shape!=(14,) or not np.isfinite(vector).all() or
                    before<max(last_end,req[index]['physics_step']) or after<=before or
                    after>last_raw_step or (index+1<n and after>req[index+1]['physics_step'])):
                raise ValueError('applied actions are not causal or mapped to requests')
            if i and index<int(f['applied/request_index'][i-1]):
                raise ValueError('applied request order regressed')
            if allow_policy_calls_on_expert_overlay and any(
                    row.get('action_index')!=i+1 or row.get('motion_phase')!='policy_action'
                    for row in physics_rows[before+1:after+1]):
                raise ValueError('expert-overlay physics does not match actual policy call')
            per_request[index]+=1;last_end=after
        if any(not count for count in per_request):
            raise ValueError('policy request has no actually applied action')
        reported_success=int(f.attrs['policy_reported_success'])
    return {'schema':'phizju_policy_rollout_audit_v1',
            'status':'candidate_self_rollout_not_expert_or_training_approved',
            'task':trace['task'],'seed':trace['seed'],'episode':trace['episode'],
            'actual_infer_attempts':n,'executed_action_count':sum(per_request),
            'causal_valid_requests':trace['causal_valid_requests'],
            'policy_reported_success':reported_success,
            'sha256':{key:sha256(path) for key,path in
                      (('raw',raw),('obs',obs),('policy',policy),('hdf5',hdf5))}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','obs','policy','hdf5'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(validate(a.raw,a.obs,a.policy,a.hdf5),indent=2))


if __name__=='__main__':main()

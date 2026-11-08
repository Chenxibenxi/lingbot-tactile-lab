"""Verify actual action grouping and timing independently of configured cadence."""
from controlled_feedback_schedule import matched_noise_seed, execution_count
import math


def audit_requests(requests, *, seed, cadence, action_request_indices,
                   action_before, action_after, official_success):
    if not requests or type(official_success) is not bool:
        raise ValueError('requests and explicit outcome required')
    if not (len(action_request_indices)==len(action_before)==len(action_after)):
        raise ValueError('applied action columns differ')
    total=len(action_before)
    if not 1<=total<=400:raise ValueError('invalid action count')
    cursor=0;clock=0
    for index,request in enumerate(requests):
        if type(request.get('dt_s')) not in (int,float) or not math.isfinite(request['dt_s']) or request['dt_s']<=0:
            raise ValueError('invalid physical timestep')
        planned=execution_count(cadence,cursor)
        if (request['request_index']!=index or request['executed_actions']!=cursor
                or request['noise_seed']!=matched_noise_seed(seed,cursor)
                or request['replan_steps']!=cadence or request['planned_action_count']!=planned
                or request['physics_step']!=clock or request['capture_physics_step']!=clock
                or request['history_anchor']!=clock):
            raise ValueError('request schedule/noise/causal anchor mismatch')
        timings=request['executed_action_timing'];count=len(timings)
        if not 1<=count<=planned:raise ValueError('empty or oversized execution segment')
        if count!=planned and not (index==len(requests)-1 and official_success):
            raise ValueError('short segment without successful terminal')
        for offset,timing in enumerate(timings):
            pos=cursor+offset
            if pos>=total or action_request_indices[pos]!=index:
                raise ValueError('action attached to wrong request')
            before,after=action_before[pos],action_after[pos]
            if before!=clock or after<=before:
                raise ValueError('noncontiguous applied physics')
            if timing['physics_step_before']!=before or timing['physics_step_after']!=after:
                raise ValueError('timing differs from recorded applied action')
            for name,step in (('history_age_at_action_start_s',before),('history_age_at_action_end_s',after)):
                expected=(step-request['history_anchor'])*request['dt_s']
                if type(timing[name]) not in (int,float) or not math.isfinite(timing[name]) or abs(timing[name]-expected)>1e-9:
                    raise ValueError('wrong history age')
            clock=after
        cursor+=count
    if cursor!=total:raise ValueError('uncovered actions')
    return dict(actual_cadence_verified=True,requests=len(requests),actions=total,
                physics_steps=clock,training_approval=False)


def audit_hdf5(path,requests,*,seed,cadence,official_success):
    import h5py
    with h5py.File(path,'r') as h:
        return audit_requests(requests,seed=seed,cadence=cadence,official_success=official_success,
            action_request_indices=h['applied/request_index'][:].tolist(),
            action_before=h['applied/physics_step_before'][:].tolist(),
            action_after=h['applied/physics_step_after'][:].tolist())

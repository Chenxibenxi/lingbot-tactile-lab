"""Weak auxiliary contact targets from SAME-RUN, AFTER-action physics ONLY.

These are future labels, never model inputs or calibrated gripper safety limits.
A single successful self-rollout is not an approved tactile training set.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import h5py
from validate_expert_data import read_jsonl
from validate_policy_rollout import validate as validate_rollout


def executed_interval_targets(physics_rows, intervals, *, dt_s):
    """Aggregate each executed action interval (before, after]; no interpolation."""
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError('positive finite physical dt required')
    rows=iter(physics_rows)
    current=next(rows,None)
    result=[];last_end=-1
    for request,index,before,after in intervals:
        if not isinstance(request,int) or not isinstance(index,int) or not (0<=before<after) or before<last_end:
            raise ValueError('invalid or overlapping executed-action interval')
        while current is not None and current['physics_step']<=before:
            current=next(rows,None)
        contact={'left':False,'right':False}
        first={'left':None,'right':None}
        peak={'left':0.,'right':0.}
        observed=0
        for step in range(before+1,after+1):
            if current is None or current['physics_step']!=step:
                raise ValueError('missing physics telemetry inside applied action')
            for side in ('left','right'):
                side_row=current[side]
                points=side_row['points'];proxy=float(side_row['load_proxy_n'])
                if type(points)!=int or points<0 or not math.isfinite(proxy) or proxy<0:
                    raise ValueError('invalid per-side contact/load telemetry')
                if points>0 and not contact[side]:
                    first[side]=step
                contact[side] |= points>0
                peak[side]=max(peak[side],proxy)
            observed+=1
            current=next(rows,None)
        result.append({'request_index':request,'action_index_in_request':index,
                       'physics_step_before':before,'physics_step_after':after,
                       'source':'future_physics_after_executed_action_not_model_input',
                       'left_contact':contact['left'],'right_contact':contact['right'],
                       'left_first_step_in_interval':first['left'],
                       'right_first_step_in_interval':first['right'],
                       'left_peak_impulse_dt_proxy_n':peak['left'],
                       'right_peak_impulse_dt_proxy_n':peak['right'],
                       'physical_samples':observed,'duration_s':round((after-before)*dt_s,8)})
        last_end=after
    return result


def derive(raw,obs,policy,hdf5):
    audit=validate_rollout(raw,obs,policy,hdf5)
    header,*physics=read_jsonl(raw)
    with h5py.File(hdf5,'r') as f:
        intervals=[(int(r),int(i),int(b),int(a)) for r,i,b,a in zip(
            f['applied/request_index'][:],f['applied/index_in_request'][:],
            f['applied/physics_step_before'][:],f['applied/physics_step_after'][:])]
    labels=executed_interval_targets(physics,intervals,dt_s=float(header['dt_s']))
    return {'schema':'phizju_self_rollout_contact_targets_v1',
            'status':'research_auxiliary_candidates_not_training_approved',
            'task':audit['task'],'seed':audit['seed'],'episode':audit['episode'],
            'policy_reported_success':audit['policy_reported_success'],
            'executed_actions':len(labels),
            'left_contact_actions':sum(int(r['left_contact']) for r in labels),
            'right_contact_actions':sum(int(r['right_contact']) for r in labels),
            'labels':labels}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','obs','policy','hdf5'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--out',type=Path,help='private research file; defaults to summary only')
    a=p.parse_args();result=derive(a.raw,a.obs,a.policy,a.hdf5)
    if a.out:
        with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='labels'},indent=2))


if __name__=='__main__':main()

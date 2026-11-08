"""Validate passive TOPP provenance; planner exceptions remain visible diagnostics."""
import math
import numpy as np


def audit(events, requests, actions):
    timing=[]
    for request in requests:
        for action in request['executed_action_timing']:
            timing.append((request['request_index'], action['physics_step_before']))
    if len(timing)!=actions or len(events)!=actions*4:
        raise ValueError('exactly two paired TOPP calls per executed action required')
    fallbacks=0;maximum=0;duration=0.
    for i in range(actions):
        for side_index,side in enumerate(('left','right')):
            start,end=events[i*4+side_index*2:i*4+side_index*2+2]
            call=side_index
            context=dict(action_index=i,request_index=timing[i][0],physics_step=timing[i][1])
            if (start.get('event')!='topp_start' or end.get('event') not in ('topp_return','topp_exception')
                    or any(x.get('call')!=call or x.get('side')!=side or x.get('context')!=context for x in (start,end))):
                raise ValueError('TOPP identity/order/context mismatch')
            path=np.asarray(start['path'],dtype=float)
            if path.ndim!=2 or path.shape[0]!=2 or path.shape[1]<1 or list(path.shape)!=start['path_shape'] or not np.isfinite(path).all() or start['step']!=.004:
                raise ValueError('invalid original qpos path/step')
            wall=end.get('wall_seconds')
            if type(wall) not in (int,float) or not math.isfinite(wall) or wall<0:
                raise ValueError('invalid planner wall time')
            if end['event']=='topp_exception':
                if not end.get('exception') or not isinstance(end.get('error'),str):raise ValueError('missing original exception')
                fallbacks+=1;continue
            n=end.get('samples');d=end.get('duration_seconds')
            if type(n) is not int or n<0 or type(d) not in (int,float) or not math.isfinite(d) or d<0:
                raise ValueError('invalid TOPP sample/duration')
            fallbacks+=int(n==0);maximum=max(maximum,n);duration+=d
    return dict(complete=True,actions=actions,calls=actions*2,fallback_calls=fallbacks,
                maximum_planned_samples=maximum,total_planned_duration_seconds=duration,
                limitation='Planned samples are not necessarily executed samples; success can stop an action early.')

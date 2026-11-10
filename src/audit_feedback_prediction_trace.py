"""Cross-check the passive trace against independently stored executed commands."""
import json
from pathlib import Path
import h5py
import numpy as np

def audit(trial):
    trial=Path(trial);rows=[json.loads(l) for l in (trial/'prediction_trace.jsonl').read_text().splitlines()]
    if rows[0].get('schema')!='feedback_prediction_trace_v1' or rows[-1]!=dict(event='terminal',complete=True):raise ValueError('complete diagnostic trace required')
    result=json.loads((trial/'result.json').read_text())
    if result.get('status')!='completed' or result.get('resource_censored') is not False:raise ValueError('uncensored episode required')
    def physical(x):
        for side in ('left','right'):
            v=x[side]
            for k in ('arm_actual_qpos','arm_drive_targets'):
                a=np.asarray(v[k]);assert a.shape==(6,) and np.isfinite(a).all()
            q=np.asarray(v['entity_actual_qpos']);d=v['active_joint_dofs']
            assert len(q)==sum(d) and len(d)==len(v['active_joint_names']) and np.isfinite(q).all()
    with h5py.File(trial/'rollout.hdf5') as h:
        command=h['applied/command_14'][:];before=h['applied/physics_step_before'][:];after=h['applied/physics_step_after'][:]
        request=h['applied/request_index'][:];index=h['applied/index_in_request'][:];step=h['request/physics_step'][:]
        nr=na=0;block=None
        for row in rows[1:-1]:
            if row['event']=='prediction':
                assert row['index']==nr and row['physics_step']==int(step[nr])
                block=np.asarray(row['commands']);assert block.shape==(50,14) and np.isfinite(block).all();physical(row['physical_state']);nr+=1
            elif row['event']=='action':
                assert row['request_index']==nr-1==int(request[na]) and row['index_in_request']==int(index[na])
                assert row['physics_before']==int(before[na]) and row['physics_after']==int(after[na])
                assert np.array_equal(np.asarray(row['command'],dtype=command.dtype),command[na])
                assert np.array_equal(block[int(index[na])],np.asarray(row['command']))
                physical(row['before']);physical(row['after']);na+=1
            else:raise ValueError('unexpected event')
        assert na==len(command)==result['actions'] and nr==len(step)==len(result['requests'])
    return dict(trace_complete=True,requests=nr,actions=na,executed_prefix_matches=True,physical_fields_finite=True,training_approval=False)

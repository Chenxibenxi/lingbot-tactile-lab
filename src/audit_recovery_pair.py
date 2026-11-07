"""Initial-source and pre-intervention equality for the fixed development pair."""
import json
from pathlib import Path
import h5py
import numpy as np
from validate_expert_data import read_jsonl,sha256


def audit(control,recovery,seed):
    control,recovery=Path(control),Path(recovery)
    if type(seed) is not int or seed not in range(100119,100123):raise ValueError('fixed development seed required')
    for p,condition in [(control,'expert_control'),(recovery,'left_open_once_recovery')]:
        if not (p/'result.json').exists():return dict(technical_comparable=False,reason='missing completed development result')
        result=json.loads((p/'result.json').read_text())
        if result.get('seed')!=seed or result.get('condition')!=condition or result.get('status')!='completed':
            raise ValueError('development result identity/status mismatch')
        if sha256(p/'action_journal.jsonl')!=result.get('action_journal_sha256'):
            raise ValueError('action journal changed since completion audit')
        raw=[q for q in (p/'force').glob('*.jsonl') if not q.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl'))]
        if len(raw)!=1:raise ValueError('one raw source stream required')
        paths={'raw':raw[0],'obs':raw[0].with_suffix('.obs.jsonl'),
               'frames':raw[0].with_suffix('.frames.jsonl'),'hdf5':p/'demonstration/data/episode0.hdf5'}
        hashes=result.get('source_audit',{}).get('sha256',{})
        if set(hashes)!=set(paths) or any(sha256(q)!=hashes[k] for k,q in paths.items()):
            raise ValueError('raw/HDF5 sources changed since completion audit')
    paths=[p/'demonstration/data/episode0.hdf5' for p in (control,recovery)]
    if not all(p.exists() for p in paths):return dict(technical_comparable=False,reason='missing HDF5')
    checks={}
    with h5py.File(paths[0]) as left,h5py.File(paths[1]) as right:
        keys=['joint_action/vector']+[f'observation/{cam}/rgb' for cam in ('head_camera','left_camera','right_camera')]
        if any(k not in h or len(h[k])==0 for h in (left,right) for k in keys):
            return dict(technical_comparable=False,reason='missing initial observations')
        checks['initial_rgb_state_equal']=all(np.array_equal(left[k][0],right[k][0]) for k in keys)
    rows=[]
    for p in (control,recovery):
        events=read_jsonl(p/'action_journal.jsonl')
        before=[x['action_index'] for x in events if x.get('event')=='before_action']
        after=[x for x in events if x.get('event')=='after_action']
        if before!=list(range(len(after))) or [x['action_index'] for x in after]!=before:
            return dict(technical_comparable=False,reason='missing/duplicate/unresolved action marker')
        rows.append(after)
    interventions=[i for i,x in enumerate(rows[1]) if x['source_role']=='external_intervention']
    checks['control_no_intervention']=all(x['source_role']=='expert' for x in rows[0])
    checks['at_most_one_recovery_intervention']=len(interventions)<=1
    checks['recovery_roles_known']=all(x['source_role'] in ('expert','external_intervention') for x in rows[1])
    prefix=interventions[0] if len(interventions)==1 else None
    # A complete attempt that never reaches the intervention still belongs in
    # the fixed denominator. Check its common unperturbed prefix, without
    # claiming that a recovery intervention occurred or succeeded.
    compared=prefix if prefix is not None else min(len(rows[0]),len(rows[1]))
    checks['pre_intervention_commands_equal']=bool(compared>0 and len(rows[0])>=compared and all(
        np.array_equal(rows[0][i]['command_state_14'],rows[1][i]['command_state_14'])
        and rows[0][i]['before_physics_step']==rows[1][i]['before_physics_step']
        and rows[0][i]['after_physics_step']==rows[1][i]['after_physics_step'] for i in range(compared)))
    checks['same_assets']=json.loads((control/'scenario.json').read_text())==json.loads((recovery/'scenario.json').read_text())
    return dict(technical_comparable=all(checks.values()),checks=checks,intervention_action_index=prefix,
                intervention_applied=len(interventions)==1,compared_prefix_actions=compared,
                training_approval=False)

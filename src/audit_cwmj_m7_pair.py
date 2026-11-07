"""Technical pairing gate for one seed/repetition; missing data is not equality."""
import json
from pathlib import Path
import h5py,numpy as np
from m7_tactile_controls import CONDITIONS
def audit_pair(trials,expected_seed=None,expected_repeat=None):
    results={a:json.loads((Path(trials[a])/'result.json').read_text()) for a in CONDITIONS}
    for a,r in results.items():
        if r.get('arm')!=a:raise ValueError('result condition mismatch')
        if r.get('status')=='completed':
            if type(r.get('success')) is not bool:raise ValueError('missing/nonboolean terminal success')
            if r.get('termination')=='task_success' and not r['success']:raise ValueError('inconsistent success termination')
            if r['success'] and r.get('termination')!='task_success':raise ValueError('success without task-success termination')
        if expected_seed is not None and r.get('seed')!=expected_seed:raise ValueError('result seed mismatch')
        if expected_repeat is not None and r.get('status')=='completed' and r.get('repeat_index')!=expected_repeat:raise ValueError('result repetition mismatch')
    full={a:r.get('status')=='completed' and r.get('termination') in ('task_success','official_action_limit') for a,r in results.items()}
    report={'full_episodes':full,'outcomes':{a:r.get('success') for a,r in results.items()},'technical_comparable':False,'checks':{}}
    if any(not (Path(trials[a])/'rollout.hdf5').exists() for a in CONDITIONS):report['reason']='missing rollout after exception';return report
    handles={}
    try:
        handles={a:h5py.File(Path(trials[a])/'rollout.hdf5') for a in CONDITIONS};ref=handles['zero']
        if any(not len(h['state_14']) or len(h['applied/command_14'])<10 for h in handles.values()):report['reason']='insufficient initial observations/actions';return report
        c=report['checks'];c['initial_rgb_state_equal']=all(np.array_equal(ref[k][0],h[k][0]) for h in handles.values() for k in ['images/head','images/left','images/right','state_14'])
        c['first10_commands_equal']=all(np.array_equal(ref['applied/command_14'][:10],h['applied/command_14'][:10]) for h in handles.values())
        c['initial_invalid_all']=all(not bool(h['tactile/window_valid'][0,0]) for h in handles.values())
        scenarios=[json.loads((Path(trials[a])/'scenario.json').read_text()) for a in CONDITIONS];c['same_assets']=all(s==scenarios[0] for s in scenarios)
        # M7 has four separately trained conditions; precontact equality after
        # the first invalid window is not a valid invariant across their weights.
        report['technical_comparable']=all(c.values());report['eligible_complete_pair']=all(full.values()) and report['technical_comparable']
        return report
    finally:
        for h in handles.values():h.close()

"""Reuse technical within-node checks, require M8 node identity explicitly."""
import json
from pathlib import Path
import h5py
import numpy as np
from audit_cwmj_m7_pair import audit_pair as audit_four


def audit_pair(trials, seed, repeat, steps):
    if steps not in (720,2880):raise ValueError('fixed duration required')
    for trial in trials.values():
        r=json.loads((Path(trial)/'result.json').read_text())
        if r.get('status')=='completed' and r.get('training_steps')!=steps:
            raise ValueError('wrong duration result')
    return audit_four(trials, expected_seed=seed, expected_repeat=repeat)


def audit_initial_across_nodes_and_repeats(trials):
    """24 complete arms for one seed, no silent omission of a node/repetition."""
    if len(trials)!=24 or len(set(map(str,trials)))!=24:
        raise ValueError('24 distinct trial paths required')
    expected={(r,s,a) for r in range(3) for s in (720,2880) for a in ('zero','true','mismatch','contact_state')}
    results=[json.loads((Path(p)/'result.json').read_text()) for p in trials]
    if { (r.get('repeat_index'),r.get('training_steps'),r.get('arm')) for r in results }!=expected:
        raise ValueError('incomplete cross-node identities')
    if len({r['seed'] for r in results})!=1:raise ValueError('mixed seeds')
    keys=['images/head','images/left','images/right','state_14']
    with h5py.File(Path(trials[0])/'rollout.hdf5') as ref:
        reference={k:ref[k][0] for k in keys}
        commands=ref['applied/command_14'][:10]
        if len(commands)!=10:raise ValueError('incomplete reference actions')
    for trial in trials[1:]:
        with h5py.File(Path(trial)/'rollout.hdf5') as h:
            if any(not np.array_equal(reference[k],h[k][0]) for k in keys):return False
            if not np.array_equal(commands,h['applied/command_14'][:10]):return False
    return True

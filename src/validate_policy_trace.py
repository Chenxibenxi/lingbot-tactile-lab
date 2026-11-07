"""Audit same-run passive tactile timeline at ACTUAL policy infer attempts.

This only validates physics/camera/request joins. It NEVER certifies training
samples: RGB pixels in an HDF5 file, executed action labels and a
hash-bound successful completion manifest are absent from these sidecars.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from align_temporal import align
from validate_expert_data import read_jsonl


def validate(raw_path, obs_path, policy_path, *, allow_policy_calls_on_expert_overlay=False):
    raw, obs, policy = map(Path, (raw_path, obs_path, policy_path))
    rh, oh, ph = (read_jsonl(p)[0] for p in (raw, obs, policy))
    if (rh.get('schema'), oh.get('schema'), ph.get('schema')) != (
            'phizju_force_probe_v1', 'phizju_obs_marker_v1', 'phizju_policy_request_v1'):
        raise ValueError('wrong passive policy trace schemas')
    if oh.get('probe_file') != raw.name or ph.get('probe_file') != raw.name:
        raise ValueError('sidecars do not refer to the same physics stream')
    if rh.get('clock_origin') != 'first_instrumented_policy_action_step_after_setup':
        rows=read_jsonl(raw)[1:]
        if (not allow_policy_calls_on_expert_overlay or
                rh.get('clock_origin')!='first_instrumented_expert_step_after_setup_stability' or
                not rows or any(r.get('motion_phase')!='policy_action' for r in rows)):
            raise ValueError('unexpected expert or unknown policy clock origin')
    for field in ('task', 'seed', 'episode'):
        if rh[field] != oh[field] or rh[field] != ph[field]:
            raise ValueError(f'{field} identity mismatch')
    dt=float(rh['dt_s'])
    if any(not math.isclose(float(h['dt_s']),dt,abs_tol=1e-7) for h in (oh,ph)):
        raise ValueError('physics timestep mismatch')
    windows=list(align(raw, obs))
    markers=read_jsonl(obs)[1:]
    requests=read_jsonl(policy)[1:]
    if not requests:
        raise ValueError('no actual client infer attempts')
    if len(markers)!=len(windows):
        raise ValueError('camera markers and causal windows differ')
    by_observation={row['obs_index']:row for row in markers}
    by_window={row['observation_marker_index']:row for row in windows}
    last_step=-1
    for index,row in enumerate(requests):
        obs_index=row.get('obs_index')
        marker=by_observation.get(obs_index)
        window=by_window.get(obs_index)
        if (row.get('request_index')!=index or marker is None or window is None or
                row.get('physics_step')!=marker.get('physics_step') or
                row['physics_step']<=last_step):
            raise ValueError('missing, stale, duplicate or reordered policy request')
        if row.get('request_monotonic_ns',-1)<marker.get('capture_monotonic_ns',0):
            raise ValueError('infer marker precedes its camera capture')
        if not math.isclose(float(row['sim_time_s']),round(row['physics_step']*dt,8),abs_tol=2e-7):
            raise ValueError('policy request time drift')
        source_step=window['quality']['last_source_step']
        if source_step is not None and source_step>row['physics_step']:
            raise ValueError('future physics source in policy input')
        last_step=row['physics_step']
    return {'schema':'phizju_policy_trace_audit_v1',
            'status':'diagnostic_only_not_training_approved',
            'task':rh['task'],'seed':rh['seed'],'episode':rh['episode'],
            'camera_captures':len(markers),'actual_infer_attempts':len(requests),
            'causal_valid_requests':sum(by_window[r['obs_index']]['quality']['window_valid'] for r in requests),
            'last_request_step':last_step,
            'missing_training_artifacts':['rgb_hdf5', 'executed_action_labels', 'successful_completion_manifest']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','obs','policy'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(validate(a.raw,a.obs,a.policy),indent=2))


if __name__=='__main__':main()

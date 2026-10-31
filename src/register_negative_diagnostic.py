"""Register relay expert REJECTIONS as read-only diagnostics, never training.

A failed planner with saved pickle camera markers is NOT an RGB HDF5 expert
sample, and an unstable scene with no physics is NOT a grasp failure. No
pickle is deserialized, no HDF5 generated, no positive completion written.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import socket
from pathlib import Path
from validate_expert_data import read_jsonl


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_negative_card(trial,repo):
    repo=Path(repo).resolve(strict=True);trial=Path(trial).resolve(strict=True)
    if trial.parent!=repo/'data/paired':raise ValueError('negative trial must originate in relay paired root')
    failure_path=trial/'failure.json';failure=json.loads(failure_path.read_text())
    if failure.get('schema')!='phizju_expert_trial_rejected_v1' or failure.get('accepted_for_training') is not False:
        raise ValueError('only explicit rejected expert trials are diagnostics')
    if failure.get('source_machine_role')!='relay_single_rtx3090' or failure.get('source_hostname')!=socket.gethostname():
        raise ValueError('not a local relay 3090 rejection')
    if failure.get('planner_ok') is not False or failure.get('probe_failed') is not False:
        raise ValueError('this diagnostic requires actual planner failure and healthy probe')
    for name in ('completion.json','expert_success_gate.json'):
        if (trial/name).exists():raise ValueError('success markers forbidden in negative-only diagnostic')
    if (trial/'demonstration/data'/f"episode{failure['episode']}.hdf5").exists():
        raise ValueError('materialized HDF5 must be assessed separately, not promoted through negative registry')
    basename=failure.get('raw_basename')
    if not isinstance(basename,str) or Path(basename).name!=basename or not basename.endswith('.jsonl'):
        raise ValueError('unsafe/missing raw force basename')
    raw=trial/'force'/basename;obs=raw.with_suffix('.obs.jsonl');frames=raw.with_suffix('.frames.jsonl')
    header,*physics=read_jsonl(raw);oh,*observations=read_jsonl(obs);fh,*markers=read_jsonl(frames)
    for metadata in (header,oh,fh):
        if any(metadata.get(k)!=failure.get(k) for k in ('task','seed','episode')):
            raise ValueError('failure/source identity mismatch')
    if any(x['physics_step']!=i for i,x in enumerate(physics,1)):
        raise ValueError('noncontiguous rejected trial physics')
    if len(markers)!=len(observations) or len(markers)!=failure.get('saved_frame_markers'):
        raise ValueError('failure frame count does not match same-run markers')
    for index,(frame,observation) in enumerate(zip(markers,observations)):
        if (frame.get('frame_index')!=index or frame.get('obs_index')!=observation.get('obs_index') or
                frame.get('physics_step')!=observation.get('physics_step') or
                not 0<=frame.get('physics_step',-1)<=len(physics)):
            raise ValueError('bad rejected frame/observation join')
    finger_counts={}
    for side in ('left','right'):
        if any(type(row[side]['points'])!=int or row[side]['points']<0 for row in physics):
            raise ValueError('bad per-side contact points')
        finger_counts[side]=sum(int(row[side]['points']>0) for row in physics)
    return {'schema':'lingbot_tactile_rejected_trial_diagnostic_v1',
            'status':'quarantined_diagnostic_not_training_or_eval',
            'source_machine_role':'relay_single_rtx3090',
            'trial_id':trial.name,'task':failure['task'],'seed':failure['seed'],
            'rejection_type':'planner_failed_before_task_goal',
            'planner_ok':False,'goal_ok':failure.get('goal_ok'),
            'physics_steps':len(physics),'saved_pickle_frame_markers':len(markers),
            'contact_physics_steps_by_side':finger_counts,
            'has_validated_rgb_hdf5':False,'same_run_observation_markers':True,
            'suitable_as_grasp_failure_example':False,
            'accepted_for_training':False,
            'source_sha256':{name:sha256(path) for name,path in
                             (('failure',failure_path),('raw',raw),('obs',obs),('frames',frames))},
            'note':'No HDF5 and no successful expert gate. Never merge into positive candidate cards or infer a failed grasp from a planner rejection.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path);p.add_argument('--trial',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    if args.out.exists():p.error('refuse overwrite')
    card=build_negative_card(args.trial,args.repo)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as file:json.dump(card,file,indent=2);file.write('\n')
    print('QUARANTINED_NEGATIVE_DIAGNOSTIC',card['trial_id'],card['physics_steps'],card['saved_pickle_frame_markers'])


if __name__=='__main__':main()

"""CPU-only command-label informativeness audit on verified TRAIN seeds.

Do not fit model parameters or inspect validation/test command labels. Target
is next-frame 14D expert command, never a calibrated force/success signal.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_samples import PairedEpisode
from plan_relay_training import build_plan


def diagnose(repo,split_path,*,instruction='Lift the pot with both arms',change_eps=1e-5):
    repo=Path(repo).resolve(strict=True);split_path=Path(split_path).resolve(strict=True)
    if repo/'datasets' not in split_path.parents or not np.isfinite(change_eps) or change_eps<0:
        raise ValueError('invalid candidate split path or command epsilon')
    split=json.loads(split_path.read_text())
    current=build_plan(repo,sorted((repo/'datasets/cards').glob('*.json')),
                       task='lift_pot',instruction=instruction)['split']
    if split!=current or split['status']!='candidate_not_approved' or split['evaluation_claims_approved']:
        raise ValueError('candidate split/card integrity mismatch')
    by_trial={};step_intervals=Counter()
    for trial in split['train']:
        ep=PairedEpisode(repo,trial,instruction=instruction)
        strata={'past_contact':[],'no_past_contact':[]}
        for i in range(len(ep)):
            row=ep.sample(i,include_images=False);inp=row['input'];target=row['target']
            h=inp['tactile_history'];m=inp['tactile_mask']
            command=inp['command_state_14'];next_command=target['action_command_14']
            if h.shape!=(5,4,3) or m.shape!=(5,4) or command.shape!=(14,) or next_command.shape!=(14,):
                raise ValueError('invalid train command/causal window geometry')
            if not np.isfinite(h).all() or not np.isfinite(m).all() or not np.isfinite(command).all() or not np.isfinite(next_command).all():
                raise ValueError('nonfinite train command/window')
            dt=row['metadata']['next_frame_step_delta']
            if not isinstance(dt,int) or dt<=0:raise ValueError('noncausal next-frame step')
            step_intervals[dt]+=1
            label='past_contact' if np.any((h[...,:2]>0)&(m[...,None]>0)) else 'no_past_contact'
            delta=np.abs(next_command.astype(np.float64)-command.astype(np.float64))
            strata[label].append((float(delta.mean()),bool(np.any(delta>change_eps))))
        by_trial[trial]={}
        for label,rows in strata.items():
            if not rows:raise ValueError('training seed has no samples in one contact stratum')
            by_trial[trial][label]={'n_pairs':len(rows),'changed_command_pairs':sum(int(x[1]) for x in rows),
                                    'mean_abs_command_delta_14d':float(np.mean([x[0] for x in rows]))}
    if not by_trial:raise ValueError('empty training split')
    return {'schema':'lingbot_tactile_train_label_diagnostic_v1',
            'status':'train_fold_diagnostic_only_no_model_fit',
            'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'source_train_trials':list(split['train']),
            'change_epsilon_command_units':float(change_eps),
            'frame_physics_step_delta_counts':{str(k):v for k,v in sorted(step_intervals.items())},
            'by_trial_and_observed_past_contact':by_trial,
            'caution':'14D dimensions mix joint/gripper command scales; this is label-change descriptives, not force or task success'}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('refuse overwrite')
    result=diagnose(args.repo,args.split)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps({'status':result['status'],'train_trials':len(result['source_train_trials']),
                      'frame_delta_counts':result['frame_physics_step_delta_counts']},indent=2))


if __name__=='__main__':main()

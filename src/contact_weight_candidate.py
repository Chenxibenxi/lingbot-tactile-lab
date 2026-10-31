"""Candidate train-fold, seed-macro balanced contact-window loss weights.

No model fitting or approval. Weights use only observed *past* contact on
verified train trials, never future force or any held-out sample value.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_samples import PairedEpisode
from plan_relay_training import build_plan


def build_weights(repo,split_path,*,instruction='Lift the pot with both arms'):
    repo=Path(repo).resolve(strict=True);split_path=Path(split_path).resolve(strict=True)
    if repo/'datasets' not in split_path.parents:
        raise ValueError('candidate split outside relay registry')
    split=json.loads(split_path.read_text())
    current=build_plan(repo,sorted((repo/'datasets/cards').glob('*.json')),
                       task='lift_pot',instruction=instruction)['split']
    if split!=current or split['status']!='candidate_not_approved' or split['evaluation_claims_approved']:
        raise ValueError('candidate split/card integrity mismatch')
    train=split['train'];total=0;seed_counts={}
    for trial in train:
        episode=PairedEpisode(repo,trial,instruction=instruction)
        counts={'past_contact':0,'no_past_contact':0}
        for idx in range(len(episode)):
            sample=episode.sample(idx,include_images=False)
            history=sample['input']['tactile_history'];mask=sample['input']['tactile_mask']
            if history.shape!=(5,4,3) or mask.shape!=(5,4) or not np.isfinite(history).all() or not np.isfinite(mask).all():
                raise ValueError('invalid causal train tactile input')
            if sample['metadata']['next_frame_step_delta']<=0 or not sample['input']['tactile_window_valid']:
                raise ValueError('invalid train source/physics delta')
            label='past_contact' if np.any((history[...,:2]>0)&(mask[...,None]>0)) else 'no_past_contact'
            counts[label]+=1
        if min(counts.values())==0:
            raise ValueError(f'{trial}: need both observed contact and no-contact training examples')
        seed_counts[trial]=counts;total+=sum(counts.values())
    if not train:raise ValueError('empty train split')
    # Equal total contribution per seed and per *observed* contact stratum.
    # Sum of weights over all train samples equals number of train samples.
    weights={trial:{label:total/(len(train)*2*count) for label,count in counts.items()}
             for trial,counts in seed_counts.items()}
    weighted_sum=sum(counts[label]*weights[trial][label]
                     for trial,counts in seed_counts.items() for label in counts)
    if not np.isclose(weighted_sum,total,atol=1e-9):raise AssertionError('weight sum mismatch')
    return {'schema':'lingbot_tactile_train_contact_weights_v1',
            'status':'candidate_not_approved','training_approval':False,
            'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'source_train_trials':train,'n_train_pairs':total,
            'observed_past_contact_counts_by_trial':seed_counts,
            'loss_weight_by_trial_and_past_contact':weights,
            'rule':'equal aggregate loss by independent train seed and observed past-contact class; mean weight=1; no validation/test fitting',
            'risk':'reweights sparse, expert-success-selected samples; may overfit and cannot alone prove tactile benefit'}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('refusing to overwrite candidate weights')
    result=build_weights(args.repo,args.split)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps({'status':result['status'],'n_train_pairs':result['n_train_pairs'],
                      'counts':result['observed_past_contact_counts_by_trial']},indent=2))


if __name__=='__main__':main()

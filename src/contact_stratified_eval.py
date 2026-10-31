"""Offline, pre-registered per-contact-window imitation error scorer.

Accepts ONLY complete, seed-matched predictions. Actual targets come from
verified relay same-run HDF5 sidecars, never from prediction files. This is
NOT robot task success, closed-loop performance, or evidence of tactile gain.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_samples import PairedEpisode
from plan_relay_training import build_plan

MODES=('visual_only','tactile','permuted_causal_history')
JOINTS=(0,1,2,3,4,5,7,8,9,10,11,12)
GRIPPERS=(6,13)


def evaluate(repo,split_path,predictions,*,instruction='Lift the pot with both arms'):
    repo=Path(repo).resolve(strict=True);split_path=Path(split_path).resolve(strict=True)
    if repo/'datasets' not in split_path.parents:
        raise ValueError('split outside relay dataset registry')
    split=json.loads(split_path.read_text());cards=sorted((repo/'datasets/cards').glob('*.json'))
    current=build_plan(repo,cards,task='lift_pot',instruction=instruction)['split']
    if split!=current or split['status']!='candidate_not_approved' or split['evaluation_claims_approved']:
        raise ValueError('split/card integrity or candidate status mismatch')
    split_sha=hashlib.sha256(split_path.read_bytes()).hexdigest()
    if predictions.get('schema')!='lingbot_tactile_predictions_v1' or predictions.get('split_sha256')!=split_sha:
        raise ValueError('prediction schema or pinned split SHA mismatch')
    vision_sha=predictions.get('vision_checkpoint_sha256')
    if not isinstance(vision_sha,str) or len(vision_sha)!=64 or any(c not in '0123456789abcdef' for c in vision_sha):
        raise ValueError('vision checkpoint SHA required for controlled comparison')
    if set(predictions.get('model_adapter_sha256_by_mode',{}))!=set(MODES):
        raise ValueError('all model provenance entries are mandatory')
    for mode,sha in predictions['model_adapter_sha256_by_mode'].items():
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError(f'invalid model SHA for {mode}')
    source={};fold_of={}
    for fold in ('train','validation','test'):
        for trial in split[fold]:
            ep=PairedEpisode(repo,trial,instruction=instruction)
            for i in range(len(ep)):
                row=ep.sample(i,include_images=False)
                key=(trial,row['metadata']['frame_index'])
                if key in source:raise ValueError('duplicate source frame')
                h=row['input']['tactile_history'];mask=row['input']['tactile_mask']
                contact=bool(np.any((h[...,:2]>0)&(mask[...,None]>0)))
                source[key]=(row['target']['action_command_14'],
                             'past_contact' if contact else 'no_past_contact')
                fold_of[key]=fold
    expected={(key,mode) for key in source for mode in MODES}
    seen={}
    rows=predictions.get('rows')
    if not isinstance(rows,list):raise ValueError('prediction rows must be a list')
    for row in rows:
        if set(row)!=set(('trial_id','frame_index','mode','action_14')):
            raise ValueError('prediction row has unknown/missing fields')
        key=(row['trial_id'],row['frame_index']); mode=row['mode'];entry=(key,mode)
        if entry not in expected or entry in seen:
            raise ValueError('unexpected, duplicate or out-of-fold prediction')
        action=np.asarray(row['action_14'],dtype=np.float64)
        if action.shape!=(14,) or not np.isfinite(action).all():
            raise ValueError('predicted 14D action must be finite')
        seen[entry]=action
    if set(seen)!=expected:
        raise ValueError(f'incomplete matched predictions: got {len(seen)} expected {len(expected)}')
    buckets={}
    for key,(target,stratum) in source.items():
        fold=fold_of[key];trial=key[0];target=np.asarray(target,dtype=np.float64)
        for mode in MODES:
            error=np.abs(seen[(key,mode)]-target)
            bucket=buckets.setdefault((fold,stratum,trial,mode),[])
            bucket.append([float(error.mean()),float(error[list(JOINTS)].mean()),
                           float(error[list(GRIPPERS)].mean())])
    out={'schema':'lingbot_tactile_contact_stratified_eval_v1',
         'status':'offline_imitation_error_only_not_task_success',
         'split_sha256':split_sha,'vision_checkpoint_sha256':vision_sha,
         'metrics':{}}
    for fold in ('train','validation','test'):
        out['metrics'][fold]={}
        for stratum in ('no_past_contact','past_contact'):
            by_mode={}
            for mode in MODES:
                trials=[np.asarray(values,dtype=float) for (f,s,t,m),values in buckets.items()
                        if f==fold and s==stratum and m==mode]
                if trials:
                    macro=np.asarray([t.mean(axis=0) for t in trials]).mean(axis=0)
                    by_mode[mode]={'independent_seeds_with_samples':len(trials),
                                   'n_pairs':sum(len(t) for t in trials),
                                   'macro_seed_mae_14d':float(macro[0]),
                                   'macro_seed_joint_mae_12d':float(macro[1]),
                                   'macro_seed_gripper_mae_2d':float(macro[2])}
                else:
                    by_mode[mode]={'independent_seeds_with_samples':0,'n_pairs':0,
                                   'macro_seed_mae_14d':None,
                                   'macro_seed_joint_mae_12d':None,
                                   'macro_seed_gripper_mae_2d':None}
            out['metrics'][fold][stratum]=by_mode
    return out


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True,
                   help='local JSON manifest: 3 complete mode predictions for each source pair')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('refusing to overwrite an existing score')
    predictions=json.loads(args.predictions.read_text())
    result=evaluate(args.repo,args.split,predictions)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print('OFFLINE_CAUSAL_CONTACT_STRATA_SCORED; NOT A ROBOT TASK SUCCESS CLAIM')


if __name__=='__main__':main()

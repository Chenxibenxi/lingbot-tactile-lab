"""TRAIN-only audit of expert NEXT-frame contact-target support.

Future physics contact is an offline target/diagnostic, NEVER a current model
input. A single onset per expert seed is not sufficient supervised signal.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_samples import PairedEpisode
from plan_relay_training import build_plan
from contact_targets import executed_interval_targets
from validate_expert_data import read_jsonl


INSTRUCTIONS={'lift_pot':'Lift the pot with both arms',
              'pick_diverse_bottles':'Pick and place the two diverse bottles'}


def audit_train(repo,split_path,*,task='lift_pot',instruction=None):
    if task not in INSTRUCTIONS:raise ValueError('unsupported relay expert contact task')
    if instruction is None:instruction=INSTRUCTIONS[task]
    repo=Path(repo).resolve(strict=True);split_path=Path(split_path).resolve(strict=True)
    if repo/'datasets' not in split_path.parents:
        raise ValueError('candidate split outside relay registry')
    frozen=json.loads(split_path.read_text())
    current=build_plan(repo,sorted((repo/'datasets/cards').glob('*.json')),
                       task=task,instruction=instruction)['split']
    if frozen!=current or frozen['status']!='candidate_not_approved' or frozen['evaluation_claims_approved']:
        raise ValueError('candidate split/card integrity mismatch')
    result={'schema':'lingbot_expert_future_contact_support_audit_v1',
            'status':'TRAIN_ONLY_DIAGNOSTIC_NO_AUX_TRAINING_APPROVAL',
            'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'source_train_trials':list(frozen['train']),'by_trial':{},
            'meaning':'future interval contact is a label only; current input is causal past history; no hardware force limit'}
    for trial in frozen['train']:
        episode=PairedEpisode(repo,trial,instruction=instruction)
        header,*physics=read_jsonl(episode.raw)
        intervals=[];past=[]
        for i in range(len(episode)):
            sample=episode.sample(i,include_images=False)
            x=sample['input'];h=x['tactile_history'];m=x['tactile_mask']
            if h.shape!=(5,4,3) or m.shape!=(5,4) or not np.isfinite(h).all() or not np.isfinite(m).all():
                raise ValueError('bad causal history for auxiliary-target audit')
            before=x['capture_physics_step'];delta=sample['metadata']['next_frame_step_delta']
            if type(before)!=int or type(delta)!=int or before<0 or delta<=0:
                raise ValueError('invalid future target physics interval')
            intervals.append((0,i,before,before+delta))
            past.append({'left':bool(np.any((h[:,:2,:2]>0)&(m[:,:2,None]>0))),
                         'right':bool(np.any((h[:,2:,:2]>0)&(m[:,2:,None]>0)))})
        future=executed_interval_targets(physics,intervals,dt_s=float(header['dt_s']))
        if len(future)!=len(past):raise ValueError('future label count mismatch')
        counts={'n_pairs':len(past),'both_sides_future_any':0,
                'left_future_any':0,'right_future_any':0,
                'left_new_onset_from_no_prior':0,'right_new_onset_from_no_prior':0,
                'both_new_onset_same_interval':0}
        for prior,label in zip(past,future):
            newcomers={}
            for side in ('left','right'):
                present=bool(label[side+'_contact'])
                counts[side+'_future_any']+=int(present)
                newcomers[side]=present and not prior[side]
                counts[side+'_new_onset_from_no_prior']+=int(newcomers[side])
            counts['both_sides_future_any']+=int(label['left_contact'] and label['right_contact'])
            counts['both_new_onset_same_interval']+=int(newcomers['left'] and newcomers['right'])
        result['by_trial'][trial]=counts
    return result


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--task',choices=tuple(INSTRUCTIONS),default='lift_pot')
    p.add_argument('--instruction',default=None)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('refuse overwrite')
    result=audit_train(args.repo,args.split,task=args.task,instruction=args.instruction)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps({'status':result['status'],'by_trial':result['by_trial']},indent=2))


if __name__=='__main__':main()

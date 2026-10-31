"""CPU-only candidate split/norm from VERIFIED relay expert cards, no model fitting.

Never consumes policy self-rollout, LXC, an unverified trial or test-fold stats.
Minimum five distinct actual seeds per task is only a technical split gate,
not statistical power for a scientific success-rate claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from paired_samples import PairedEpisode
from contact_normalization import fit_positive_contact_stats
from register_dataset import build_card
from split_and_norm import make_seed_splits,fit_train_only_stats


def build_plan(repo,card_paths,*,task,instruction,min_seeds=5):
    repo=Path(repo).resolve(strict=True)
    paths=list(map(Path,card_paths))
    if not paths:
        raise ValueError('no relay expert cards')
    cards=[];hashes={}
    for path in paths:
        path=path.resolve(strict=True)
        if repo/'datasets/cards' not in path.parents:
            raise ValueError('card outside relay expert registry')
        card=json.loads(path.read_text())
        # A second task may have candidate cards without entering this task's
        # normalization or folds. Only the requested task is eligible here.
        if card.get('task')!=task:
            continue
        if card.get('status')!='candidate_unassigned_split':
            raise ValueError('noncandidate expert card for requested task')
        if build_card(repo/card['data_path_relative'],repo)!=card:
            raise ValueError('card differs from current relay trial bytes')
        cards.append(card)
        hashes[card['trial_id']]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not cards:
        raise ValueError(f'no verified relay expert cards for task {task}')
    split=make_seed_splits(cards,min_seeds_per_task=min_seeds)
    split['expert_card_sha256']=hashes
    split['evaluation_claims_approved']=False
    train_ids=split['train']
    counts={}
    def training_samples():
        for trial in train_ids:
            episode=PairedEpisode(repo,trial,instruction=instruction)
            counts[trial]=len(episode)
            for i in range(len(episode)):
                yield episode.sample(i,include_images=False)
    norm=fit_train_only_stats(training_samples(),train_ids)
    contact_norm=fit_positive_contact_stats(training_samples(),train_ids)
    norm['training_approval']=False
    split_hash=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()
    norm['split_candidate_sha256']=split_hash
    contact_norm['split_candidate_sha256']=split_hash
    return {'split':split,'norm':norm,'contact_norm':contact_norm,'train_pairs_by_trial':counts,
            'status':'candidate_only_no_training_or_generalization_claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--task',default='lift_pot')
    p.add_argument('--instruction',default='Lift the pot with both arms')
    p.add_argument('--out-split',type=Path,required=True)
    p.add_argument('--out-norm',type=Path,required=True)
    p.add_argument('--out-contact-norm',type=Path,
                   help='also emit unapproved, zero-preserving positive-bin candidate')
    args=p.parse_args()
    cards=sorted((args.repo/'datasets/cards').glob('*.json'))
    result=build_plan(args.repo,cards,task=args.task,instruction=args.instruction)
    outputs=[(args.out_split,result['split']),(args.out_norm,result['norm'])]
    if args.out_contact_norm:outputs.append((args.out_contact_norm,result['contact_norm']))
    if len({path.resolve() for path,_ in outputs})!=len(outputs):
        p.error('output paths must be different')
    for path,_ in outputs:
        if path.exists():
            p.error(f'refusing to overwrite {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
    for path,value in outputs:
        with path.open('x') as file:
            json.dump(value,file,indent=2);file.write('\n')
    print(json.dumps({'status':result['status'], 'task':args.task,
                      'train_trials':len(result['split']['train']),
                      'validation_trials':len(result['split']['validation']),
                      'test_trials':len(result['split']['test']),
                      'train_pairs_by_trial':result['train_pairs_by_trial']},indent=2))


if __name__=='__main__':main()

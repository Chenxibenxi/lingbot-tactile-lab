"""Candidate v2: append two new bottle EXPERT seeds to train, freeze v1 val/test.

Train-only v1/v2 statistics are recomputed; no model is trained or approved.
Changing heldout seeds would invalidate comparability of the frozen pilot.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from paired_samples import PairedEpisode
from contact_normalization import fit_positive_contact_stats
from split_and_norm import fit_train_only_stats
from register_dataset import build_card as rebuild_expert_card
from validate_expert_data import sha256

NEW_IDS=('relay_pickdiverse_s100013_onsetpilot_expert0','relay_pickdiverse_s100015_onsetpilot_expert0')
EXPECTED_SEEDS=(100013,100015)
TASK='pick_diverse_bottles'
INSTRUCTION='Pick and place the two diverse bottles'


def extend_preserving_heldout(v1,new_cards,new_card_sha,*,parent_sha):
    if (v1.get('schema')!='lingbot_tactile_seed_split_plan_v1' or
            v1.get('status')!='candidate_not_approved' or
            v1.get('source_machine_role')!='relay_single_rtx3090' or
            v1.get('evaluation_claims_approved') is not False or
            len(v1.get('train',[]))!=3 or len(v1.get('validation',[]))!=1 or len(v1.get('test',[]))!=1 or
            set(v1.get('expert_card_sha256',{}))!=set(v1['train']+v1['validation']+v1['test'])):
        raise ValueError('frozen pilot v1 split not trusted')
    if tuple(c.get('trial_id') for c in new_cards)!=NEW_IDS or tuple(c.get('seed') for c in new_cards)!=EXPECTED_SEEDS:
        raise ValueError('only preregistered new physical seeds and expert IDs accepted')
    if set(NEW_IDS)&set(v1['train']+v1['validation']+v1['test']):
        raise ValueError('new expert already assigned in frozen v1 split')
    for c in new_cards:
        if (c.get('schema')!='lingbot_tactile_dataset_card_v1' or
                c.get('task')!=TASK or c.get('status')!='candidate_unassigned_split' or
                c.get('source_machine_role')!='relay_single_rtx3090'):
            raise ValueError('new trial is not a verified relay expert candidate')
    old_groups=v1.get('task_seed_groups',{}).get(TASK,{})
    if (set(old_groups.get('train',[]))&set(old_groups.get('validation',[])+old_groups.get('test',[])) or
            set(old_groups.get('validation',[]))&set(old_groups.get('test',[])) or
            set(EXPECTED_SEEDS)&set().union(*(old_groups.get(f,[]) for f in ('train','validation','test')))):
        raise ValueError('physical seed crosses frozen v1 folds')
    v2=deepcopy(v1)
    v2['train']=sorted(v1['train']+list(NEW_IDS))
    v2['task_seed_groups'][TASK]['train']=sorted(old_groups['train']+list(EXPECTED_SEEDS))
    v2['expert_card_sha256'].update(new_card_sha)
    v2['parent_v1_split_sha256']=parent_sha
    v2['new_unassigned_parent_experts']=list(NEW_IDS)
    v2['evaluation_claims_approved']=False
    if v2['validation']!=v1['validation'] or v2['test']!=v1['test'] or len(v2['train'])!=5:
        raise ValueError('heldout fold changed or expected five training seeds missing')
    return v2


def build(repo):
    repo=Path(repo).resolve(strict=True)
    v1_path=repo/'datasets/relay_pickdiverse_seed_split_candidate_v1.json'
    v1=json.loads(v1_path.read_text())
    index=json.loads((repo/'datasets/relay_paired_training_index_candidate_v1.json').read_text())
    if (index['tasks'][TASK]['split_sha256']!=sha256(v1_path) or
            json.loads((repo/'datasets/provenance_v1.json').read_text()).get('approved_paired_episodes')!=[]):
        raise ValueError('frozen pilot v1 source changed or training unexpectedly approved')
    cards={};hashes={}
    for trial_id in v1['train']+v1['validation']+v1['test']+list(NEW_IDS):
        card_path=repo/'datasets/cards'/f'{trial_id}.json';card=json.loads(card_path.read_text())
        if (card['task']!=TASK or card['trial_id']!=trial_id or
                rebuild_expert_card(repo/'data/paired'/trial_id,repo)!=card):
            raise ValueError('expert card no longer matches local relay HDF5/raw')
        cards[trial_id]=card;hashes[trial_id]=sha256(card_path)
        if trial_id in v1['expert_card_sha256'] and hashes[trial_id]!=v1['expert_card_sha256'][trial_id]:
            raise ValueError('frozen v1 expert card SHA changed')
    v2=extend_preserving_heldout(v1,[cards[i] for i in NEW_IDS],
                                  {i:hashes[i] for i in NEW_IDS},parent_sha=sha256(v1_path))
    episodes={trial:PairedEpisode(repo,trial,instruction=INSTRUCTION) for trial in v2['train']}
    def train_samples():
        for trial in v2['train']:
            ep=episodes[trial]
            for i in range(len(ep)):
                yield ep.sample(i,include_images=False)
    norm=fit_train_only_stats(train_samples(),v2['train'])
    positive_norm=fit_positive_contact_stats(train_samples(),v2['train'])
    norm['training_approval']=False
    candidate_hash=hashlib.sha256(json.dumps(v2,sort_keys=True).encode()).hexdigest()
    norm['split_candidate_sha256']=candidate_hash
    positive_norm['split_candidate_sha256']=candidate_hash
    new_counts={trial:len(episodes[trial]) for trial in NEW_IDS}
    if sum(new_counts.values())!=249:raise ValueError('new exact same-run causal pair count differs')
    return {'split':v2,'norm':norm,'positive_norm':positive_norm,
            'new_trial_pair_counts':new_counts,
            'total_v2_train_pairs':sum(len(ep) for ep in episodes.values()),
            'unchanged_heldout_ids':{'validation':list(v1['validation']),'test':list(v1['test'])},
            'training_approval':False,'status':'new_seeds_train_only_candidate_not_approved'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--norm',type=Path,required=True)
    p.add_argument('--positive-norm',type=Path,required=True)
    a=p.parse_args();repo=a.repo.resolve(strict=True)
    outputs=[(a.split.resolve(), 'datasets'),(a.norm.resolve(),'configs'),(a.positive_norm.resolve(),'configs')]
    if len({x for x,_ in outputs})!=3 or any(path.exists() or path.parent!=repo/subdir for path,subdir in outputs):
        p.error('all output files must be new under expected repo datasets/configs')
    result=build(repo)
    for path,key in zip((a.split,a.norm,a.positive_norm),('split','norm','positive_norm')):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as file:json.dump(result[key],file,indent=2);file.write('\n')
    print('BOTTLE_SPLIT_V2_CANDIDATE',result['total_v2_train_pairs'],result['new_trial_pair_counts'],
          result['unchanged_heldout_ids'],'training_approval',result['training_approval'])


if __name__=='__main__':main()

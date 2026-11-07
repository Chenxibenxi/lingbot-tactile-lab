"""Audit exactly two preregistered new relay expert attempts, NO split edits.

Future contact is counted only as an offline label; it is never included in
current RGB/state/past force inputs or promoted to training approval.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from contact_targets import executed_interval_targets
from paired_samples import PairedEpisode
from register_dataset import build_card as rebuild_expert_card
from validate_expert_data import read_jsonl,sha256

SEEDS=(100013,100015)
SPLIT='relay_pickdiverse_seed_split_candidate_v1.json'
INSTRUCTION='Pick and place the two diverse bottles'


def onset_counts(episode):
    header,*physics=read_jsonl(episode.raw)
    intervals=[];past=[];observed_contact=0
    for i in range(len(episode)):
        s=episode.sample(i,include_images=False);x=s['input'];h=x['tactile_history'];m=x['tactile_mask']
        if (not x['tactile_window_valid'] or h.shape!=(5,4,3) or m.shape!=(5,4) or
                not np.isfinite(h).all() or not np.isfinite(m).all()):
            raise ValueError('invalid causal current tactile input')
        start=x['capture_physics_step'];end=start+s['metadata']['next_frame_step_delta']
        if type(start)!=int or type(end)!=int or not 0<=start<end<=episode.card['physics_steps']:
            raise ValueError('future label interval is outside this expert episode')
        intervals.append((0,i,start,end))
        past.append({side:bool(np.any((h[:,lo:lo+2,:2]>0)&(m[:,lo:lo+2,None]>0)))
                     for side,lo in (('left',0),('right',2))})
        observed_contact+=int(np.any((h[...,:2]>0)&(m[...,None]>0)))
    future=executed_interval_targets(physics,intervals,dt_s=float(header['dt_s']))
    if len(future)!=len(past):raise ValueError('future interval label count differs')
    onsets={side:sum(int(item[side+'_contact'] and not prior[side])
                     for item,prior in zip(future,past)) for side in ('left','right')}
    return {'causal_pairs':len(episode),'pairs_with_past_contact':observed_contact,
            'future_new_onset_intervals_by_side':onsets}


def audit(repo):
    repo=Path(repo).resolve(strict=True)
    split_path=repo/'datasets'/SPLIT;split=json.loads(split_path.read_text())
    if split.get('status')!='candidate_not_approved' or split.get('evaluation_claims_approved') is not False:
        raise ValueError('frozen old bottle split no longer candidate')
    if json.loads((repo/'datasets/provenance_v1.json').read_text()).get('approved_paired_episodes')!=[]:
        raise ValueError('new pilot cannot silently become approved training data')
    frozen=set(split['train']+split['validation']+split['test'])
    entries=[]
    for seed in SEEDS:
        trial_id=f'relay_pickdiverse_s{seed}_onsetpilot_expert0'
        status=repo/'data/paired'/f'{trial_id}.status'
        raw=status.read_text().strip()
        if not raw.startswith('exit_code=') or not raw[10:].isdigit():
            raise ValueError('missing/invalid preregistered attempt status')
        code=int(raw[10:]);trial=repo/'data/paired'/trial_id
        if trial_id in frozen or seed in set().union(*(split['task_seed_groups']['pick_diverse_bottles'][f] for f in ('train','validation','test'))):
            raise ValueError('new seed cannot enter frozen old expert fold')
        item={'seed':seed,'trial_id':trial_id,'attempt_status_sha256':sha256(status),
              'source_machine_role':'relay_single_rtx3090','writer_exit_code':code}
        card_path=repo/'datasets/cards'/f'{trial_id}.json'
        if code==0:
            card=json.loads(card_path.read_text())
            if (card!=rebuild_expert_card(trial,repo) or card['seed']!=seed or
                    card['task']!='pick_diverse_bottles' or card['status']!='candidate_unassigned_split'):
                raise ValueError('new expert success/sha/source differs')
            episode=PairedEpisode(repo,trial_id,instruction=INSTRUCTION)
            if episode.card!=card:raise ValueError('causal paired samples differ from same-run expert')
            item.update({'status':'verified_expert_candidate_unassigned',
                         'expert_card_sha256':sha256(card_path),
                         'saved_frames':card['saved_frames'],
                         **onset_counts(episode)})
        else:
            if card_path.exists() or (trial/'completion.json').exists():
                raise ValueError('failed attempt must not masquerade as expert success')
            item.update({'status':'failed_attempt_isolated_no_expert_labels',
                         'failure_record_present':(trial/'failure.json').is_file()})
        entries.append(item)
    return {'schema':'lingbot_bottle_onset_expansion_audit_v1',
            'status':'two_fixed_attempts_source_checked_not_training_approved',
            'old_candidate_split_sha256':sha256(split_path),
            'predeclared_seeds':list(SEEDS),'attempts':entries,
            'successful_new_expert_candidates':sum(x['writer_exit_code']==0 for x in entries),
            'new_causal_pairs_unassigned':sum(x.get('causal_pairs',0) for x in entries),
            'training_approval':False,
            'note':'Original v1 seed split remains unchanged. These attempts are neither model fitting nor independent validation/test results.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():p.error('refuse existing result card')
    if a.out.resolve().parent!=(a.repo.resolve()/'datasets/expansion_diagnostics'):
        p.error('sanitized result card must live under datasets/expansion_diagnostics')
    result=audit(a.repo);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as fp:json.dump(result,fp,indent=2);fp.write('\n')
    print('BOTTLE_ONSET_TWO_FIXED_ATTEMPTS',[(x['seed'],x['status']) for x in result['attempts']])


if __name__=='__main__':main()

"""Candidate ONLY: build same-run relay expert image/action/touch sample index.

Private row-level index stays in ignored data/reports. Publish only aggregate
counts and SHA; this does not authorize training or contaminate heldout seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_samples import PairedEpisode,CAMERAS
from register_dataset import build_card as rebuild_expert_card
from validate_expert_data import sha256

TASKS={
    'lift_pot':('relay_lift_pot_seed_split_candidate_v1.json',
                'relay_lift_pot_train_norm_candidate_v1.json','Lift the pot with both arms'),
    'pick_diverse_bottles':('relay_pickdiverse_seed_split_candidate_v1.json',
                'relay_pickdiverse_train_norm_candidate_v1.json','Pick and place the two diverse bottles'),
}


def build(repo):
    repo=Path(repo).resolve(strict=True)
    registry=json.loads((repo/'datasets/provenance_v1.json').read_text())
    if registry.get('schema')!='lingbot_tactile_dataset_registry_v1' or registry.get('approved_paired_episodes')!=[]:
        raise ValueError('this index is exclusively an unapproved exploratory candidate')
    records=[];summary={};seen=set();assigned={}
    for task,(split_name,norm_name,instruction) in TASKS.items():
        split_path=repo/'datasets'/split_name;norm_path=repo/'configs'/norm_name
        split=json.loads(split_path.read_text());norm=json.loads(norm_path.read_text())
        if (split.get('schema')!='lingbot_tactile_seed_split_plan_v1' or
                split.get('status')!='candidate_not_approved' or
                split.get('source_machine_role')!='relay_single_rtx3090' or
                split.get('evaluation_claims_approved') is not False or
                norm.get('schema')!='lingbot_tactile_train_norm_v1' or
                norm.get('training_approval') is not False or
                set(norm.get('source_trials',[]))!=set(split.get('train',[])) or
                norm.get('split_candidate_sha256')!=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()):
            raise ValueError('split/norm metadata do not bind one candidate task/fold')
        folds=split.get('task_seed_groups',{}).get(task)
        if not isinstance(folds,dict):raise ValueError('actual physical seed groups missing')
        expected=set().union(*(folds.get(f,[]) for f in ('train','validation','test')))
        if any(set(folds[a])&set(folds[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))):
            raise ValueError('actual physical seed leaks between folds')
        counts={f:{'trials':0,'pairs':0,'pairs_with_observed_contact':0,'decoded_camera_frames':0}
                for f in ('train','validation','test')}
        actual=set()
        for fold in ('train','validation','test'):
            ids=split.get(fold)
            if not isinstance(ids,list) or not ids:raise ValueError('empty/non-list candidate fold')
            for trial_id in ids:
                if trial_id in seen or not isinstance(trial_id,str):raise ValueError('duplicate/malformed expert trial')
                seen.add(trial_id)
                if '_expert' not in trial_id:raise ValueError('negative or self-rollout cannot be expert supervision')
                card_path=repo/'datasets/cards'/f'{trial_id}.json'
                card=json.loads(card_path.read_text());trial=repo/'data/paired'/trial_id
                if (card.get('task')!=task or card.get('trial_id')!=trial_id or
                        card.get('seed') not in folds[fold] or
                        sha256(card_path)!=split.get('expert_card_sha256',{}).get(trial_id) or
                        rebuild_expert_card(trial,repo)!=card):
                    raise ValueError('expert source/seed/fold/HDF5 SHA not revalidated')
                group=(task,card['seed'])
                if assigned.get(group,fold)!=fold:raise ValueError('same actual seed crosses folds')
                assigned[group]=fold;actual.add(card['seed'])
                episode=PairedEpisode(repo,trial_id,instruction=instruction)
                if episode.card!=card:raise ValueError('paired data diverged from expert card')
                counts[fold]['trials']+=1
                for i in range(len(episode)):
                    sample=episode.sample(i,include_images=True)
                    meta=sample['metadata'];inp=sample['input']
                    if meta['task']!=task or meta['seed']!=card['seed'] or meta['trial_id']!=trial_id:
                        raise ValueError('paired sample provenance changed')
                    if (not inp['tactile_window_valid'] or inp['capture_physics_step']<0 or
                            meta['next_frame_step_delta']<=0 or
                            set(inp['images_rgb'])!=set(CAMERAS) or
                            any(im.shape!=(240,320,3) or im.dtype!=np.uint8 for im in inp['images_rgb'].values())):
                        raise ValueError('tactile history/time or RGB sample invalid')
                    if inp['capture_physics_step']+meta['next_frame_step_delta']>card['physics_steps']:
                        raise ValueError('target action source beyond same episode')
                    history=inp['tactile_history'];mask=inp['tactile_mask']
                    contact=bool(np.any(np.any(history[...,:2]>0,axis=-1)&(mask>0)))
                    key=(trial_id,meta['frame_index'])
                    if key in seen:raise ValueError('duplicate image/action pair')
                    seen.add(key)
                    records.append({'schema':'lingbot_private_relay_pair_index_v1',
                        'task':task,'trial_id':trial_id,'seed':card['seed'],'fold':fold,
                        'current_frame_index':meta['frame_index'],
                        'next_expert_frame_index':meta['target_frame_index'],
                        'capture_physics_step':inp['capture_physics_step'],
                        'next_expert_physics_step':inp['capture_physics_step']+meta['next_frame_step_delta'],
                        'history_nonzero_contact':contact,
                        'target_semantics':'next_saved_frame_expert_command_14_not_future_force_input'})
                    counts[fold]['pairs']+=1;counts[fold]['pairs_with_observed_contact']+=int(contact)
                    counts[fold]['decoded_camera_frames']+=len(CAMERAS)
        if actual!=expected:raise ValueError('split actual seed registry differs from all source cards')
        summary[task]={'folds':counts,'split_sha256':sha256(split_path),
                       'train_only_normalization_sha256':sha256(norm_path),
                       'expert_card_sha256':{trial:split['expert_card_sha256'][trial]
                                            for f in ('train','validation','test') for trial in split[f]}}
    return records,{'schema':'lingbot_relay_paired_training_index_candidate_v1',
        'status':'candidate_source_verified_not_training_approved',
        'source_machine_role':'relay_single_rtx3090',
        'pairs':len(records),'tasks':summary,
        'provenance_registry_sha256':sha256(repo/'datasets/provenance_v1.json'),
        'negative_or_self_rollout_pairs':0,'training_approval':False,
        'note':'Row-level private index is not a model fit, evaluation prediction, force calibration, or permission to use heldout samples for training.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--private-out',type=Path,required=True)
    p.add_argument('--card-out',type=Path,required=True);a=p.parse_args()
    repo=a.repo.resolve(strict=True);private=a.private_out.resolve();card=a.card_out.resolve()
    if private.parent!=repo/'data/reports' or card.parent!=repo/'datasets' or private.exists() or card.exists():
        p.error('fresh output paths must be private data/reports and sanitized datasets roots')
    rows,summary=build(repo)
    private.parent.mkdir(parents=True,exist_ok=True)
    with private.open('x',encoding='utf-8') as fp:
        for row in rows:fp.write(json.dumps(row,allow_nan=False)+'\n')
    summary['private_index_sha256']=sha256(private)
    with card.open('x',encoding='utf-8') as fp:json.dump(summary,fp,indent=2);fp.write('\n')
    print('PAIRED_INDEX_CANDIDATE_ONLY',summary['pairs'],
          {task:{f:data['pairs'] for f,data in body['folds'].items()} for task,body in summary['tasks'].items()})


if __name__=='__main__':main()

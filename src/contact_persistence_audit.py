"""CPU-only matched-seed contact persistence audit, not causal model training.

Cross-validates one verified expert and one deliberately modified negative; it
never pairs post-intervention frames by ordinal index or uploads RGB bytes.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from register_dataset import build_card as build_expert_card
from validate_controlled_negative import build_card as build_negative_card
from validate_expert_data import read_jsonl


def touch_runs(physics,side):
    if side not in ('left','right') or not physics:
        raise ValueError('nonempty physics and valid side required')
    runs=[];current=0;first=None;last=None;total=0
    for expected,row in enumerate(physics,1):
        if row.get('physics_step')!=expected:
            raise ValueError('physical step gap or duplicate')
        points=row.get(side,{}).get('points')
        if type(points)!=int or points<0:
            raise ValueError('invalid contact points')
        if points>0:
            total+=1;current+=1
            if first is None:first=expected
            last=expected
        elif current:
            runs.append(current);current=0
    if current:runs.append(current)
    return {'first_step':first,'last_step':last,'total_contact_steps':total,
            'runs':len(runs),'longest_contiguous_run':max(runs,default=0),
            'final_physics_step_in_contact':last==len(physics)}


def audit(repo,positive_id='relay_pickdiverse_s100000_expert0',negative_id='relay_pickdiverse_s100000_open_grasp1'):
    import cv2,h5py
    repo=Path(repo).resolve(strict=True)
    pos=repo/'data/paired'/positive_id;neg=repo/'data/negative'/negative_id
    pc=build_expert_card(pos,repo);nc=build_negative_card(repo,neg)
    pcard=repo/'datasets/cards'/f'{positive_id}.json';ncard=repo/'datasets/negative_diagnostics'/f'{negative_id}.json'
    if pc!=json.loads(pcard.read_text()) or nc!=json.loads(ncard.read_text()):
        raise ValueError('source card differs from current local source bytes')
    if pc['task']!=nc['task'] or pc['seed']!=nc['seed'] or pc['seed'] not in (100000,100007):
        raise ValueError('not preregistered matched task and physical seed')
    histories={}
    for role,path in (('positive_expert',pos),('controlled_negative',neg)):
        raws=[p for p in (path/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl'))]
        if len(raws)!=1:raise ValueError('ambiguous raw physics stream')
        _,*physics=read_jsonl(raws[0]);histories[role]={side:touch_runs(physics,side) for side in ('left','right')}
    initial={}
    with h5py.File(pos/'demonstration/data/episode0.hdf5','r') as p,h5py.File(neg/'demonstration/data/episode0.hdf5','r') as n:
        for cam in ('head_camera','left_camera','right_camera'):
            x=cv2.imdecode(np.frombuffer(p[f'observation/{cam}/rgb'][0],np.uint8),cv2.IMREAD_COLOR)
            y=cv2.imdecode(np.frombuffer(n[f'observation/{cam}/rgb'][0],np.uint8),cv2.IMREAD_COLOR)
            if x is None or y is None or x.shape!=y.shape or not np.array_equal(x,y):
                raise ValueError(f'initial {cam} RGB not pixel-identical')
            initial[cam]={'identical_rgb_pixels':True,'shape':list(x.shape)}
        if not np.array_equal(p['joint_action/vector'][0],n['joint_action/vector'][0]):
            raise ValueError('initial 14D commanded state differs')
    initial['command_state_14_identical']=True
    return {'schema':'lingbot_controlled_contact_persistence_v1',
            'status':'single_counterfactual_diagnostic_not_training_or_policy_evaluation',
            'task':pc['task'],'seed':pc['seed'],'independent_seed_count':1,
            'positive_card_sha256':hashlib.sha256(pcard.read_bytes()).hexdigest(),
            'negative_card_sha256':hashlib.sha256(ncard.read_bytes()).hexdigest(),
            'first_frame':initial,'touch_runs_by_role':histories,
            'caution':'Same-seed independent runs with different actions; do not pair later frames or infer calibrated force/model performance.'}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():p.error('refuse overwrite')
    result=audit(a.repo);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print('CONTACT_PERSISTENCE_AUDIT',result['touch_runs_by_role'])


if __name__=='__main__':main()

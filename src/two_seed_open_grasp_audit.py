"""Two distinct train-seed matched simulated interventions, diagnostic ONLY.

Rebuilds each positive+negative SHA card, confirms pixel-identical initial RGB
and 14D command, never folds failures into successful expert supervision.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from contact_persistence_audit import audit as audit_pair
from validate_contact_geometry import build_card as spatial_card
from validate_controlled_negative import build_card as negative_card
from validate_expert_data import read_jsonl,sha256

TRIALS=((100000,'relay_pickdiverse_s100000_expert0','relay_pickdiverse_s100000_open_grasp1'),
        (100007,'relay_pickdiverse_s100007_expert0','relay_pickdiverse_s100007_open_grasp_spatial1'))
FINGERS={'left':('fl_link7','fl_link8'),'right':('fr_link7','fr_link8')}


def relative_contact_change_percent(reference,changed):
    if type(reference)!=int or type(changed)!=int or reference<=0 or changed<0:
        raise ValueError('nonempty reference and nonnegative observed count required')
    return round((reference-changed)*100/reference,3)  # negative means MORE contact


def require_train_only_seeds(split,chosen):
    if split.get('schema')!='lingbot_tactile_seed_split_plan_v1' or split.get('status')!='candidate_not_approved':
        raise ValueError('no matching candidate seed split')
    group=split.get('task_seed_groups',{}).get('pick_diverse_bottles',{})
    train=set(group.get('train',[]));heldout=set(group.get('validation',[]))|set(group.get('test',[]))
    chosen_seeds=set(chosen)
    if not chosen_seeds.issubset(train) or chosen_seeds&heldout or len(chosen_seeds)!=len(chosen):
        raise ValueError('negative counterpart cannot use a validation/test or duplicate seed')
    if split.get('evaluation_claims_approved') is not False:
        raise ValueError('unexpected evaluation approval')


def finger_peaks(trial):
    raws=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl'))]
    if len(raws)!=1:raise ValueError('not a unique validated raw physics stream')
    _,*rows=read_jsonl(raws[0]);out={}
    for side,names in FINGERS.items():
        for name in names:
            values=[float(row[side]['fingers'][name]['load_proxy_n']) for row in rows]
            if not values or any(not math.isfinite(v) or v<0 for v in values):
                raise ValueError('invalid finger load proxy')
            out[name]=round(max(values),6)
    return out


def audit(repo):
    repo=Path(repo).resolve(strict=True)
    split_path=repo/'datasets/relay_pickdiverse_seed_split_candidate_v1.json'
    split=json.loads(split_path.read_text());require_train_only_seeds(split,[x[0] for x in TRIALS])
    entries=[]
    for seed,positive_id,negative_id in TRIALS:
        paired=audit_pair(repo,positive_id,negative_id)
        if paired['seed']!=seed or paired['independent_seed_count']!=1 or not paired['first_frame']['command_state_14_identical']:
            raise ValueError('mismatched start or seed')
        positive=repo/'data/paired'/positive_id;negative=repo/'data/negative'/negative_id
        nc=negative_card(repo,negative)
        if not nc['planner_ok'] or nc['goal_ok'] or nc['accepted_for_expert_training'] or nc['matched_positive_seed']!=seed:
            raise ValueError('negative outcome was not the preregistered intervention')
        if (positive_id not in split['train'] or
                split.get('expert_card_sha256',{}).get(positive_id)!=paired['positive_card_sha256'] or
                negative_id in sum((split[x] for x in ('train','validation','test')),[])):
            raise ValueError('negative entered expert fold or positive split SHA/train group changed')
        source=paired['touch_runs_by_role'];p=source['positive_expert'];n=source['controlled_negative']
        contrast={side:relative_contact_change_percent(p[side]['total_contact_steps'],n[side]['total_contact_steps'])
                  for side in ('left','right')}
        spatial_sha=None
        if seed==100007:
            sc=spatial_card(repo,negative)
            path=repo/'datasets/spatial_diagnostics'/f'{negative_id}.json'
            if sc!=json.loads(path.read_text()) or sc['occlusion_validated']:
                raise ValueError('spatial diagnostic card not verified')
            spatial_sha=sha256(path)
        entries.append({'seed':seed,'positive_trial':positive_id,'negative_trial':negative_id,
            'positive_card_sha256':paired['positive_card_sha256'],
            'negative_card_sha256':paired['negative_card_sha256'],
            'spatial_card_sha256':spatial_sha,
            'initial_rgb_and_14d_exact_match':True,
            'positive_expert_contact':p,'open_grasp_failure_contact':n,
            'contact_step_decrease_percent_by_side':contrast,
            'positive_peak_proxy_by_finger':finger_peaks(positive),
            'negative_peak_proxy_by_finger':finger_peaks(negative)})
    return {'schema':'lingbot_two_seed_controlled_open_grasp_audit_v1',
            'status':'two_preregistered_train_seed_counterfactuals_diagnostic_only',
            'task':'pick_diverse_bottles','independent_seed_count':len(entries),
            'candidate_split_sha256':sha256(split_path),
            'trial_pairs':entries,'accepted_for_expert_training':False,
            'evaluation_claims_approved':False,
            'caution':'Two intentionally modified, expert-preselected simulated seeds are not a task-failure rate, calibrated force, online tactile-policy test or statistical generalization claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():p.error('refuse overwrite')
    if a.out.resolve().parent!=(a.repo.resolve()/'datasets/counterfactual_diagnostics'):
        p.error('sanitized card must stay under datasets/counterfactual_diagnostics')
    card=audit(a.repo);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as file:json.dump(card,file,indent=2);file.write('\n')
    print('TWO_SEED_DIAGNOSTIC',[(e['seed'],e['contact_step_decrease_percent_by_side']) for e in card['trial_pairs']])


if __name__=='__main__':main()

"""Read-only SHA-bound real s100023 expert H50 source audit; never approves fit.

This audits ORIGINAL raw/HDF5/online actions and computes causal H50 candidates.
It does not mark a split approved, promote a registry, read test labels or train.
"""
from __future__ import annotations
import json
from pathlib import Path
from align_temporal import align
from patch_online_expert_stride20_pilot import REPO,PRIVATE,TRIAL,generate
from policy_highlevel_chunks import METHOD,ROLE,SOURCE_SHA,select_policy_h50
from preregister_online_expert_trial import (SOURCE,TASK_SOURCE,ROBOT_SOURCE,
                                            TASK_SOURCE_SHA,ROBOT_SOURCE_SHA)
from record_online_expert_demo import audit_same_run_online_clock
from register_dataset import build_card
from validate_expert_data import read_jsonl,sha256,validate

CARD='relay_online_expert_s100023_stride20_h50_candidate_v1.json'


def audit(repo=REPO):
    repo=Path(repo).resolve(strict=True)
    if repo!=REPO or any(sha256(src)!=digest for src,digest in
                          ((SOURCE,SOURCE_SHA),(TASK_SOURCE,TASK_SOURCE_SHA),
                           (ROBOT_SOURCE,ROBOT_SOURCE_SHA))):
        raise ValueError('not the original relay or controller/task/planner source')
    generated=generate()
    if not PRIVATE.is_dir() or any((PRIVATE/name).is_symlink() or
                                   (PRIVATE/name).read_text()!=text for name,text in generated.items()):
        raise ValueError('private new-seed runner differs from pinned original source')
    trial=repo/'data/paired'/TRIAL
    if trial.is_symlink() or not trial.is_dir():raise ValueError('missing unique relay online expert trial')
    raws=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl'))]
    if len(raws)!=1:raise ValueError('not one distinct private raw physics stream')
    raw=raws[0];obs=raw.with_suffix('.obs.jsonl');frames=raw.with_suffix('.frames.jsonl')
    hdf5=trial/'demonstration/data/episode0.hdf5';completion=trial/'completion.json'
    gate=trial/'expert_success_gate.json';actions_path=trial/'online_actions.jsonl'
    files={'raw':raw,'obs':obs,'frames':frames,'hdf5':hdf5,'completion':completion,
           'success_gate':gate,'online_actions':actions_path,'pilot_result':trial/'pilot_result.json'}
    hashes={name:sha256(path) for name,path in files.items()}
    source=validate(raw,obs,frames,hdf5,completion)
    card=build_card(trial,repo)  # in-memory expert provenance; no registry write
    card_path=repo/'datasets/cards'/f'{TRIAL}.json'
    if json.loads(card_path.read_text())!=card:
        raise ValueError('unapproved expert candidate card differs from live source bytes')
    header,*actions=read_jsonl(actions_path)
    result=json.loads((trial/'pilot_result.json').read_text())
    complete=json.loads(completion.read_text());success=json.loads(gate.read_text())
    if (source['seed']!=card['seed'] or source['seed']!=100023 or
            card['status']!='candidate_unassigned_split' or
            result.get('planning_path_stride_indices')!=20 or
            result.get('actual_take_action_calls')!=59 or result.get('physics_steps')!=6808 or
            result.get('goal_ok') is not True or result.get('planner_ok') is not True or
            result.get('training_approval') is not False or
            header.get('trial_id')!=TRIAL or header.get('seed')!=100023 or
            header.get('source_sha256')!={'original_base_task':SOURCE_SHA,
                 'original_bottle_task':TASK_SOURCE_SHA,'original_robot_planner':ROBOT_SOURCE_SHA} or
            hashes['online_actions']!=complete.get('online_action_sha256') or
            hashes['online_actions']!=success.get('online_action_sha256') or
            len(actions)!=59):
        raise ValueError('new-seed online expert identity, source or call count differs')
    clock=audit_same_run_online_clock(raw,actions,6808)
    frame_rows=read_jsonl(frames)[1:]
    windows={w['observation_marker_index']:w for w in align(raw,obs)}
    prepared=[dict(row,trial_id=TRIAL,completion_sha256=hashes['completion'],
                   original_base_task_sha256=SOURCE_SHA,source_model_role=ROLE) for row in actions]
    anchors=[]
    for index in range(len(actions)-49):
        frame=frame_rows[index]
        window=windows[frame['obs_index']]
        if not window['quality']['window_valid']:continue
        candidate=select_policy_h50(prepared,{'trial_id':TRIAL,
            'completion_sha256':hashes['completion'],
            'controller_method':METHOD,'source_model_role':ROLE,
            'original_base_task_sha256':SOURCE_SHA,
            'current_rgb_hdf5_sha256':hashes['hdf5'],
            'tactile_window_valid':True,'first_action_index':index,
            'capture_physics_step':actions[index]['before_physics_step'],
            'past_tactile_max_physics_step':window['quality']['last_source_step']},
            expected_trial=TRIAL,expected_completion_sha=hashes['completion'],
            source_physics_steps=6808,fold='train')
        if candidate.get('training_approval') is not False or candidate['target_action_count']!=50:
            raise ValueError('unauthorized H50 source promotion')
        anchors.append(index)
    registry=json.loads((repo/'datasets/provenance_v1.json').read_text())
    if registry.get('approved_paired_episodes')!=[] or not anchors:
        raise ValueError('owner training approval unexpectedly changed or no H50 candidate')
    report={'schema':'lingbot_online_expert_s100023_h50_source_audit_v1',
            'status':'source_verified_train_intent_candidate_not_approved',
            'task':'pick_diverse_bottles','seed':100023,'source_machine_role':'relay_single_rtx3090',
            'stride_path_indices':20,'actual_take_action_calls':59,'physics_steps':6808,
            'saved_frames':source['saved_frames'],'planner_ok':True,'task_goal_ok':True,
            'online_highlevel_clock_verified':clock['source_checked_same_run_clock_only'],
            'causal_h50_candidate_anchors_count':len(anchors),
            'causal_h50_candidate_anchor_indices':anchors,
            'fold_intent':'train_only_not_assigned',
            'independent_validation_h50_candidates':0,
            'approved_h50_train_windows':0,'approved_h50_validation_windows':0,
            'gpu_training_permitted_now':False,'training_approval':False,
            'source_file_sha256':hashes}
    recorded=json.loads((repo/'datasets'/CARD).read_text())
    if (set(recorded)!=set(report)|{'note'} or
            any(recorded.get(key)!=value for key,value in report.items())):
        raise ValueError('sanitized H50 candidate card differs from real source audit')
    return report


if __name__=='__main__':
    value=audit()
    print(json.dumps({'status':value['status'],'seed':value['seed'],
          'real_H50_candidate_anchors':value['causal_h50_candidate_anchors_count'],
          'training_approval':value['training_approval']},sort_keys=True))

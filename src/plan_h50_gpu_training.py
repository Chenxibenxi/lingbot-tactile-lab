"""Read-only H50 GPU training readiness REPORT, never a launcher or approval.

It inspects only frozen/sanitized metadata and the one private short pilot SHA;
no RGB, weights, train labels, model fit, or GPU allocation occurs here. A
future full H50 source validator and explicit owner data approval are mandatory.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from validate_expert_data import sha256

REPO=Path(__file__).resolve().parents[1]
SHORT_TRIAL='relay_pickdiverse_s100022_online_h50_expert0'
STRIDE20_TRIAL='relay_pickdiverse_s100023_online_h50_stride20_expert0'
STRIDE20_CARD='relay_online_expert_s100023_stride20_h50_candidate_v1.json'
VAL_TRIAL='relay_pickdiverse_s100024_online_h50_stride20_validation_expert0'
VAL_REJECTION_CARD='relay_online_validation_s100024_rejected_v1.json'
UNSTABLE_VAL_TRIAL='relay_pickdiverse_s100025_online_h50_stride20_validation_expert0'
UNSTABLE_VAL_CARD='relay_online_validation_s100025_unstable_v1.json'


def verify_unstable_validation_setup(repo):
    """No goal/capture can be inferred from an unstable setup; never retry silently."""
    path=repo/'datasets'/UNSTABLE_VAL_CARD
    trial=repo/'data/paired'/UNSTABLE_VAL_TRIAL
    if not path.is_file():
        if trial.exists():raise ValueError('unstable validation trial missing rejection card')
        return False
    card=json.loads(path.read_text())
    if (card.get('schema')!='lingbot_online_validation_s100025_init_rejection_v1' or
            card.get('status')!='unstable_initial_object_before_any_highlevel_action' or
            card.get('seed')!=100025 or card.get('exception_type')!='UnStableError' or
            card.get('actual_take_action_calls')!=0 or card.get('recorded_physics_steps') is not None or
            card.get('expert_hdf5_written') is not False or
            card.get('completion_written') is not False or
            card.get('independent_validation_h50_verified')!=0 or
            card.get('training_approval') is not False):
        raise ValueError('unstable validation metadata changed')
    if not trial.is_dir() or trial.is_symlink() or (trial/'completion.json').exists() or \
            (trial/'demonstration/data/episode0.hdf5').exists() or \
            (trial/'online_actions.jsonl').exists():
        raise ValueError('unstable validation trial missing or falsely promoted')
    private=repo/'work/online_expert_stride20_s100025_validation_v1'
    files={'preregistration':trial/'preregistration.json',
           'rejection':trial/'failure.json',
           'private_preregistration_code':private/'preregister_online_expert_validation_s100025.py',
           'private_recording_code':private/'record_online_expert_validation_s100025.py'}
    if any(sha256(file)!=card['source_file_sha256'][name] for name,file in files.items()):
        raise ValueError('unstable validation private source SHA changed')
    failure=json.loads(files['rejection'].read_text())
    if (failure.get('schema')!='lingbot_online_expert_pilot_rejected_v1' or
            failure.get('seed')!=100025 or failure.get('exception_type')!='UnStableError' or
            failure.get('physics_steps') is not None or failure.get('training_approval') is not False):
        raise ValueError('unstable validation failure source changed')
    return True


def verify_rejected_validation(repo):
    """Bound the zero-val claim to the actual unsuccessful independent run."""
    path=repo/'datasets'/VAL_REJECTION_CARD
    trial=repo/'data/paired'/VAL_TRIAL
    if not path.is_file():
        if trial.exists():raise ValueError('new validation trial exists without signed negative result card')
        return False
    card=json.loads(path.read_text())
    if (card.get('schema')!='lingbot_online_validation_s100024_rejection_v1' or
            card.get('status')!='80_true_calls_but_task_goal_false_no_validation_h50' or
            card.get('seed')!=100024 or card.get('actual_take_action_calls')!=80 or
            card.get('physics_steps')!=9559 or card.get('task_goal_success') is not False or
            card.get('independent_validation_h50_verified')!=0 or
            card.get('expert_hdf5_written') is not False or
            card.get('completion_written') is not False or
            card.get('training_approval') is not False):
        raise ValueError('independent validation negative outcome card changed')
    if not trial.is_dir() or trial.is_symlink() or (trial/'completion.json').exists() or \
            (trial/'demonstration/data/episode0.hdf5').exists():
        raise ValueError('failed validation source missing or silently promoted')
    raws=[p for p in (trial/'force').glob('*.jsonl')
          if not p.name.endswith(('.obs.jsonl','.frames.jsonl'))]
    if len(raws)!=1:raise ValueError('failed validation raw physics source ambiguous')
    raw=raws[0]
    files={'raw':raw,'obs':raw.with_suffix('.obs.jsonl'),
           'frames':raw.with_suffix('.frames.jsonl'),
           'actions':trial/'online_actions.jsonl',
           'result':trial/'pilot_result.json','rejection':trial/'failure.json'}
    if any(sha256(file)!=card['source_file_sha256'][name] for name,file in files.items()):
        raise ValueError('failed validation six-source SHA changed')
    return True


def inspect(repo=REPO):
    repo=Path(repo).resolve(strict=True)
    registry=json.loads((repo/'datasets/provenance_v1.json').read_text())
    split=json.loads((repo/'datasets/relay_pickdiverse_seed_split_candidate_v2.json').read_text())
    norm=json.loads((repo/'configs/relay_pickdiverse_contact_norm_candidate_v3.json').read_text())
    visual=json.loads((repo/'datasets/relay_v2_real_visual_zero_identity_v1.json').read_text())
    short=json.loads((repo/'datasets/relay_online_expert_s100022_h50_short_v1.json').read_text())
    if (registry.get('schema')!='lingbot_tactile_dataset_registry_v1' or
            split.get('schema')!='lingbot_tactile_seed_split_plan_v1' or
            norm.get('schema')!='lingbot_tactile_contact_norm_v2' or
            visual.get('schema')!='lingbot_real_v2_visual_zero_action_identity_v1' or
            short.get('schema')!='lingbot_online_rule_expert_newseed_no_h50_v1'):
        raise ValueError('training readiness source metadata schema changed')
    if (short.get('seed')!=100022 or not short.get('task_goal_success') or
            short.get('actual_take_action_calls')!=46 or short.get('h50_required_calls')!=50 or
            short.get('eligible_h50_windows_verified')!=0 or short.get('training_approval') is not False or
            short.get('expert_hdf5_written') is not False or short.get('expert_completion_written') is not False):
        raise ValueError('recorded short online expert negative gate changed')
    trial=repo/'data/paired'/SHORT_TRIAL
    if trial.exists():
        if (trial/'completion.json').exists() or (trial/'demonstration/data/episode0.hdf5').exists():
            raise ValueError('rejected H50 pilot was silently promoted')
        actions=trial/'online_actions.jsonl'
        if sha256(actions)!=short['source_file_sha256']['actions']:
            raise ValueError('private short online expert action SHA changed')
    val_negative_verified=verify_rejected_validation(repo)
    val_unstable_verified=verify_unstable_validation_setup(repo)
    if (registry.get('approved_paired_episodes')!=[] or
            split.get('status')!='candidate_not_approved' or
            norm.get('training_approval') is not False or
            visual.get('training_approval') is not False or
            visual.get('adapter_backward_run') is not False or
            visual.get('server_rpc_run') is not False):
        raise ValueError('previously unapproved dataset/model state changed; require a new owner audit')
    if (visual.get('base_frozen') is not True or
            visual.get('visual_vs_zero_tactile_bitwise_equal') is not True or
            visual.get('zero_residual_adapter_parameters')!=65664):
        raise ValueError('real visual-zero BF16 inference proof changed')
    candidate_path=repo/'datasets'/STRIDE20_CARD
    online_completion=repo/'data/paired'/STRIDE20_TRIAL/'completion.json'
    candidate_windows=0
    if candidate_path.is_file() and online_completion.is_file():
        if repo!=REPO:raise ValueError('only local relay can revalidate private H50 source')
        from audit_online_expert_stride20 import audit
        candidate=audit(repo)  # full raw/HDF5/source SHA and causal 50-call source gate
        candidate_windows=candidate['causal_h50_candidate_anchors_count']
    return {'schema':'lingbot_h50_real_gpu_training_readiness_v1',
            'status':'not_ready_metadata_report_not_training_authority',
            'task':'pick_diverse_bottles','target_semantics':'actual_expert_take_action_h50',
            'approved_h50_train_windows':0,'approved_h50_validation_windows':0,
            'verified_unapproved_h50_train_intent_windows':candidate_windows,
            'verified_unapproved_h50_validation_windows':0,
            'independent_validation_trial_failed_goal_verified':val_negative_verified,
            'independent_validation_trial_unstable_setup_verified':val_unstable_verified,
            'approved_paired_episodes':0,
            'short_newseed_task_success_but_calls':short['actual_take_action_calls'],
            'short_newseed_excluded_from_h50':True,
            'existing_bottle_frame_pairs_are_not_h50_labels':True,
            'candidate_bottle_split_status':split['status'],
            'candidate_train_only_norm_approved':False,
            'real_bf16_154800_visual_zero_sample_actions_proved':True,
            'real_model_training_forward_proved':False,
            'real_adapter_backward_or_optimizer_proved':False,
            'live_tactile_rpc_proved':False,
            'gpu_training_is_intended_after_data_gates':True,
            'gpu_training_permitted_now':False,
            'CPU_proxy_training_run':False,
            'blockers':[('one source-verified independent H50 seed has only train intent; validation H50 still absent'
                         if candidate_windows else 'no independently source-verified online H50 train/validation seeds'),
                        'split and train-only contact normalization lack owner approval',
                        'full real VLA training forward/backward GPU memory not measured',
                        'matched visual/frozen and shuffled-tactile GPU controls not run'],
            'source_sha256':{'registry':sha256(repo/'datasets/provenance_v1.json'),
                'bottle_split_candidate':sha256(repo/'datasets/relay_pickdiverse_seed_split_candidate_v2.json'),
                'bottle_contact_norm_candidate':sha256(repo/'configs/relay_pickdiverse_contact_norm_candidate_v3.json'),
                'real_visual_zero_proof':sha256(repo/'datasets/relay_v2_real_visual_zero_identity_v1.json'),
                'short_newseed_diagnostic':sha256(repo/'datasets/relay_online_expert_s100022_h50_short_v1.json'),
                **({'unapproved_s100023_h50_candidate':sha256(candidate_path)} if candidate_windows else {}),
                **({'rejected_s100024_validation':sha256(repo/'datasets'/VAL_REJECTION_CARD)}
                   if val_negative_verified else {}),
                **({'rejected_s100025_unstable_validation':sha256(repo/'datasets'/UNSTABLE_VAL_CARD)}
                   if val_unstable_verified else {})},
            'training_approval':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=REPO)
    args=p.parse_args()
    print(json.dumps(inspect(args.repo),ensure_ascii=False,sort_keys=True,indent=2))


if __name__=='__main__':main()

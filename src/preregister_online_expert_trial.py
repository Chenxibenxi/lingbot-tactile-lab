"""CPU-only reservation for ONE independent online-planned expert H50 pilot.

Never constructs actions, starts RoboTwin, modifies a split, or approves training.
The live expert action generator must be implemented and separately verified first.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from policy_highlevel_chunks import SOURCE_SHA
from validate_expert_data import sha256

REPO=Path(__file__).resolve().parents[1]
SOURCE=Path('/mnt/relaydata/lingbot/RoboTwin/envs/_base_task.py')
TASK_SOURCE=Path('/mnt/relaydata/lingbot/RoboTwin/envs/pick_diverse_bottles.py')
ROBOT_SOURCE=Path('/mnt/relaydata/lingbot/RoboTwin/envs/robot/robot.py')
TASK_SOURCE_SHA='95ebb80fa96eafd0b7ff5518251c2a7aa42f22ef9b7cfdda4e80eac511f06151'
ROBOT_SOURCE_SHA='3dcd80acc8cab489a4c5edb507cc460dab1724be0226b8d5c4c1b218dee605cb'
SEED=100022
TASK='pick_diverse_bottles'
TRIAL='relay_pickdiverse_s100022_online_h50_expert0'


def preregister(repo=REPO, *, seed=SEED, source=SOURCE,
                task_source=TASK_SOURCE, robot_source=ROBOT_SOURCE):
    """Reserve only a seed-distinct prospective trial; no private output or GPU."""
    repo=Path(repo).resolve(strict=True)
    source=Path(source).resolve(strict=True)
    task_source=Path(task_source).resolve(strict=True)
    robot_source=Path(robot_source).resolve(strict=True)
    if type(seed) is not int or seed!=SEED or sha256(source)!=SOURCE_SHA:
        raise ValueError('new independent expert seed or original controller SHA changed')
    if sha256(task_source)!=TASK_SOURCE_SHA or sha256(robot_source)!=ROBOT_SOURCE_SHA:
        raise ValueError('online bottle task or robot path-planner source SHA changed')
    split=json.loads((repo/'datasets/relay_pickdiverse_seed_split_candidate_v2.json').read_text())
    groups=split['task_seed_groups'][TASK]
    if (split.get('status')!='candidate_not_approved' or
            set(groups)!={'train','validation','test'} or
            any(seed in groups[fold] for fold in groups) or
            set(groups['train'])&set(groups['validation']) or
            set(groups['train'])&set(groups['test']) or
            set(groups['validation'])&set(groups['test'])):
        raise ValueError('candidate split not frozen or trial seed overlaps any fold')
    root=repo/'data/paired'
    if not root.is_dir() or not re.fullmatch(r'[A-Za-z0-9_-]+',TRIAL):
        raise ValueError('missing relay paired root or invalid single trial ID')
    if (root/TRIAL).exists() or any(root.glob(f'*s{seed}*')):
        raise ValueError('fresh seed already attempted; never reuse or relabel a trial')
    registry=json.loads((repo/'datasets/provenance_v1.json').read_text())
    if registry.get('approved_paired_episodes')!=[]:
        raise ValueError('dataset approval state changed; review before expert pilot')
    return {'schema':'lingbot_online_expert_newseed_preregistration_v1',
            'status':'CPU_design_only_not_executed', 'task':TASK,'seed':seed,
            'new_trial_id':TRIAL,'source_machine_role':'relay_single_rtx3090',
            'fold_intent':'train_candidate_only_not_assigned',
            'original_base_task_sha256':SOURCE_SHA,
            'original_bottle_task_sha256':TASK_SOURCE_SHA,
            'original_robot_planner_sha256':ROBOT_SOURCE_SHA,
            'required_controller_method':'Base_Task.take_action',
            'required_action_source':'live_rule_expert_plan_from_current_state_not_old_drive_replay',
            'required_consecutive_calls':50,'future_contact_or_old_dense_steps_as_labels':False,
            'source_and_completion_HDF5_verified':False,
            'online_generator_cpu_contract_tested':True,
            'online_generator_implemented_and_validated':False,
            'gpu_run_allowed_now':False,'training_approval':False}


if __name__=='__main__':
    print(json.dumps(preregister(),sort_keys=True))

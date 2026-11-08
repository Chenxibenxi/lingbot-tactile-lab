"""Opt-in v2 collector. Requires a separately frozen implementation manifest."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil

from validate_expert_data import sha256
from recovery_capture_bridge_v2 import CaptureBridge
from recovery_pair_state_v2 import capture_state
from recovery_prefix_generation_v2 import generate_prefix, stage_commands, seal_prefix
from recovery_paired_executor_v2 import execute_pair
from recovery_source_audit_v2 import audit_sources

R = Path(__file__).resolve().parents[1]
OVERLAY = R/'work/overlay_policy_equivalent_pilot_v1'
PROTOCOL = R/'docs/RECOVERY_DEVELOPMENT_PROTOCOL_V2_ZH.md'
KINDS = ('prefix_generation','expert_control','left_open_once_recovery')


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def preflight(seed, kind):
    reservation = json.loads((R/'datasets/recovery_development_reservations_v2.json').read_text())
    if type(seed) is not int or seed not in [x['seed'] for x in reservation['seeds']] or kind not in KINDS:
        raise ValueError('registered v2 attempt required')
    if reservation['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('reserved protocol changed')
    manifest = json.loads((R/'data/reports/recovery_development_v2/manifest.json').read_text())
    if manifest.get('gpu_execution_ready') is not True or manifest.get('protocol_sha256') != sha256(PROTOCOL):
        raise ValueError('v2 implementation is not frozen and ready')
    required = {str(path.relative_to(R)) for path in (R/'src').glob('recovery_*v2.py')}
    required.add('src/record_recovery_development_v2.py')
    if not required.issubset(manifest['files']):
        raise ValueError('missing v2 source pins')
    for name, digest in manifest['files'].items():
        if sha256(R/name) != digest: raise ValueError('frozen source changed: '+name)
    if Path.cwd().resolve() != OVERLAY or shutil.disk_usage(R).free < 3*1024**3:
        raise ValueError('isolated cwd and at least 3 GiB free required')
    from patch_policy_equivalent_expert_overlay import generate, SOURCE
    from preregister_online_expert_trial import TASK_SOURCE, ROBOT_SOURCE, TASK_SOURCE_SHA, ROBOT_SOURCE_SHA
    if (OVERLAY/'envs/_base_task.py').read_text() != generate(SOURCE.read_text()):
        raise ValueError('overlay drift')
    if sha256(TASK_SOURCE) != TASK_SOURCE_SHA or sha256(ROBOT_SOURCE) != ROBOT_SOURCE_SHA:
        raise ValueError('baseline drift')
    from probe_real_v2_bf16_stream import require_gpu_window
    require_gpu_window()
    return manifest


def run(seed, kind):
    manifest = preflight(seed, kind)
    identity = dict(seed=seed, protocol_sha256=sha256(PROTOCOL),
                    asset_sha256=sha256(R/'work/cwmj_bottle_asset_v2/manifest.json'),
                    controller_sha256=sha256(OVERLAY/'envs/_base_task.py'))
    prefix_dir = R/f'data/paired/recovery_v2_{seed}_prefix_generation'
    prefix_path = prefix_dir/'prefix.json'
    seal = None
    if kind != 'prefix_generation':
        prior = json.loads((prefix_dir/'result.json').read_text())
        if prior.get('status') != 'completed' or prior.get('identity') != identity or prior.get('training_approval') is not False:
            raise ValueError('completed same-identity prefix required')
        seal = prior['prefix_seal']
        if sha256(prefix_path) != seal['sha256']:
            raise ValueError('prefix changed')
        source = prior['source_audit']['source']
        for name, rel in prior['source_paths'].items():
            if sha256(prefix_dir/rel) != source['sha256'][name]:
                raise ValueError('prefix source changed: '+name)
    out = R/f'data/paired/recovery_v2_{seed}_{kind}'; out.mkdir(exist_ok=False)
    write_new(out/'preregistration.json', manifest)
    os.environ.update(FORCE_PROBE_EXPERT_DIR=str(out/'force'), TACTILE_POLICY_EQUIVALENT_EXPERT='1',
                      TACTILE_ONLINE_EXPERT_PILOT_APPROVED='1', TACTILE_SOURCE_MACHINE_ROLE='relay_single_rtx3090')
    from record_online_expert_demo import _config
    from cwmj_bottle_factory import install
    scenario = install(); scenario['factory_sha256'] = sha256(R/'src/cwmj_bottle_factory.py')
    write_new(out/'scenario.json', scenario)
    task = None; closed = False
    with (out/'action_journal.jsonl').open('x', encoding='utf-8') as journal:
        def emit(event):
            if event.get('training_approval') is not False:
                raise ValueError('journal cannot authorize training')
            journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
            journal.flush(); os.fsync(journal.fileno())
        emit(dict(event='journal_header', identity=identity, kind=kind, training_approval=False))
        try:
            task = importlib.import_module('envs.pick_diverse_bottles').pick_diverse_bottles()
            task.setup_demo(now_ep_num=0, seed=seed, **_config(OVERLAY,out))
            bridge = CaptureBridge(task, emit=emit, image_dir=out/'snapshots')
            if kind == 'prefix_generation':
                document = generate_prefix(identity=identity, commands=lambda stage:stage_commands(task,stage),
                    read_clock=bridge.read_clock, read_state=bridge.read_state,
                    read_rgb_hashes=bridge.read_rgb_hashes, execute=bridge.execute, emit=emit)
                if task.eval_success: raise ValueError('task ended before intervention boundary')
                rows = [dict(receipt, action_index=i, phase=document['rows'][i]['phase'],
                             source_role='expert', segment_index=0) for i,receipt in enumerate(bridge.receipts)]
                execution = dict(rows=rows, physics_steps=bridge.read_clock(), official_success=False,
                                 action_count=len(rows), termination='prefix_boundary', training_approval=False)
            else:
                execution = execute_pair(task, prefix_path, expected_sha256=seal['sha256'],
                    expected_identity=identity, condition=kind, emit=emit, bridge=bridge)
            write_new(out/'execution.json', execution)
            terminal_state = bridge.read_state()
            terminal_rgb = bridge.read_rgb_hashes()
            write_new(out/'terminal_observation.json', dict(state=terminal_state, rgb_sha256=terminal_rgb,
                snapshot_index=bridge.snapshot_count-1, official_success=execution['official_success'],
                training_approval=False))
            raw = task._force_probe.raw_path
            task.close_env(); closed = True
            task.merge_pkl_to_hdf5_video()
            paths = dict(raw=raw, obs=raw.with_suffix('.obs.jsonl'), frames=raw.with_suffix('.frames.jsonl'),
                         hdf5=out/'demonstration/data/episode0.hdf5')
            audit = audit_sources(**paths, actions=execution['rows'], seed=seed,
                physics_steps=execution['physics_steps'], official_success=execution['official_success'], kind=kind)
            if kind == 'prefix_generation':
                seal = seal_prefix(prefix_path, document, expected_identity=identity)
            write_new(out/'result.json', dict(schema='recovery_development_complete_v2', status='completed',
                identity=identity, kind=kind, source_audit=audit, prefix_seal=seal,
                source_paths={key:str(path.relative_to(out)) for key,path in paths.items()},
                action_journal_sha256=sha256(out/'action_journal.jsonl'),
                official_success=execution['official_success'], training_approval=False))
        except Exception as exc:
            write_new(out/'failure.json', dict(status='needs_inspection', identity=identity, kind=kind,
                error_type=type(exc).__name__, error=str(exc), training_approval=False))
            raise
        finally:
            if task is not None and not closed:
                try: task.close_env()
                except Exception as exc:
                    write_new(out/'cleanup_failure.json', dict(error=str(exc), training_approval=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--kind', choices=KINDS, required=True)
    args = parser.parse_args()
    run(args.seed, args.kind)

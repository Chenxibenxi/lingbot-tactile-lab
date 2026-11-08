"""Same-run source checks for partial prefix and terminal branch episodes."""
from pathlib import Path
from recovery_clock_audit import audit_clock
from validate_expert_data import read_jsonl, validate, sha256


def audit_sources(*, raw, obs, frames, hdf5, actions, seed, physics_steps,
                  official_success, kind):
    if kind not in ('prefix_generation','expert_control','left_open_once_recovery'):
        raise ValueError('unknown episode kind')
    if type(seed) is not int or type(physics_steps) is not int or type(official_success) is not bool:
        raise ValueError('explicit source identity required')
    if kind == 'prefix_generation' and official_success:
        raise ValueError('partial prefix must not claim task success')
    # No legacy successful-task completion is fabricated for a partial prefix.
    source = validate(raw, obs, frames, hdf5)
    if (source['task'] != 'pick_diverse_bottles' or source['seed'] != seed
            or source['episode'] != 0 or source['physics_steps'] != physics_steps
            or source['saved_frames'] != len(actions)):
        raise ValueError('source does not cover registered execution')
    clock = audit_clock(actions, read_jsonl(raw)[1:], read_jsonl(frames)[1:],
                        read_jsonl(obs)[1:], official_success=official_success)
    interventions = clock['external_interventions']
    if (kind != 'left_open_once_recovery' and interventions != 0) or interventions > 1:
        raise ValueError('invalid intervention count')
    if kind == 'prefix_generation':
        if (not actions or actions[0].get('phase') != 'grasp' or actions[-1].get('phase') != 'lift'
                or any(row.get('phase') not in ('grasp','lift') for row in actions)):
            raise ValueError('prefix contains non-prefix action phases')
    # Detect accidental source changes during the audit before publishing receipt.
    for name, path in dict(raw=raw, obs=obs, frames=frames, hdf5=hdf5).items():
        if sha256(Path(path)) != source['sha256'][name]:
            raise ValueError('source changed during audit')
    return dict(schema='recovery_source_audit_v2', kind=kind, source=source, clock=clock,
                source_integrity_passed=True, official_success=official_success,
                training_approval=False,
                limitation='Source integrity only; requires sealed prefix and pair-state audits separately.')

"""Derived consumption snapshot; never changes folds, raw trials or frozen protocol."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from confirmation_completion_inventory import inventory, CONDITIONS

R = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root=R):
    source = root / 'datasets/tactile_seed_bank_m1_m6_v2.json'
    bank = copy.deepcopy(json.loads(source.read_text()))
    state = inventory(root)
    bank.update(schema='lingbot_tactile_seed_bank_v3',
                snapshot_utc=datetime.now(timezone.utc).isoformat(),
                parent_bank_sha256=digest(source),
                confirmation_batch_accounted=state['ready_for_final_analysis'],
                statistical_eligibility='Determined separately by final paired audit')
    found = set()
    for entry in bank['entries']:
        if entry.get('stage') != 'future_confirmation':
            continue
        seed = entry['seed']
        if seed not in range(100071, 100087) or seed in found or entry['fold'] != 'test':
            raise ValueError('Confirmation seed identity or fold mismatch')
        found.add(seed)
        attempts = []
        for repeat in range(3):
            for arm in CONDITIONS:
                name = f'r{repeat}_{seed}_{arm}'
                raw = root / f'data/eval/cwmj_m6_confirm_{name}'
                item = dict(name=name, repeat=repeat, arm=arm, raw_dir=str(raw))
                if name in state['terminal']:
                    result = state['terminal'][name]
                    item.update(status=result['status'], success=result.get('success'),
                                result_sha256=digest(raw / 'result.json'))
                elif name in state['interrupted']:
                    item.update(status='audited_interruption', success=None,
                                evidence=state['interrupted'][name])
                else:
                    item['status'] = 'started_without_terminal' if raw.exists() else 'not_started'
                attempts.append(item)
        consumed = any(a['status'] != 'not_started' for a in attempts)
        entry.update(stage='M6_confirmation', role='fixed_independent_confirmation',
                     status='consumed_test_seed' if consumed else 'reserved_not_executed',
                     available_as_new_test_seed=not consumed,
                     used_as_expert_supervision_in_m6=False,
                     excluded_due_to_interruption=seed in state['interrupted_seeds'],
                     attempts=attempts)
    if found != set(range(100071, 100087)):
        raise ValueError('Missing confirmation reservations')
    return bank


if __name__ == '__main__':
    bank = build()
    destination = R / 'datasets/tactile_seed_bank_m1_m6_v3.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(bank, indent=2) + '\n')
    temporary.replace(destination)
    print('SEED_BANK_CONSUMPTION_SNAPSHOT', destination)

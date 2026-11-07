"""Read-only final-analysis prerequisite; interruptions never become failures."""
import hashlib
import json
from pathlib import Path

CONDITIONS = ('zero', 'true', 'mismatch', 'true_input_zero')


def inventory(root, seeds=range(100071, 100087), repeats=range(3)):
    root = Path(root)
    planned = {f'r{r}_{s}_{a}': (s, r, a)
               for s in seeds for r in repeats for a in CONDITIONS}
    interrupted = {}
    for path in (root / 'data/reports').glob('cwmj_m6_confirmation_interruption_*.json'):
        record = json.loads(path.read_text())
        name = record['interrupted_attempt']
        if name not in planned or name in interrupted:
            raise ValueError('Unknown or duplicate interrupted attempt')
        hashes = record['original_files_sha256']
        if not hashes:
            raise ValueError('Missing interruption evidence')
        expected = (root / f'data/eval/cwmj_m6_confirm_{name}').resolve()
        for relative, digest in hashes.items():
            source = (root / relative).resolve()
            if not source.is_relative_to(expected):
                raise ValueError('Interruption evidence outside attempt')
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('Interrupted evidence changed')
        interrupted[name] = record
    terminal, pending = {}, []
    for name, (seed, repeat, arm) in planned.items():
        path = root / f'data/eval/cwmj_m6_confirm_{name}/result.json'
        if name in interrupted:
            if path.exists():
                raise ValueError('Interrupted attempt unexpectedly has result')
            continue
        if not path.exists():
            pending.append(name)
            continue
        result = json.loads(path.read_text())
        if result.get('seed') != seed or result.get('arm') != arm:
            raise ValueError('Terminal identity mismatch')
        if result.get('status') == 'completed':
            if result.get('repeat_index') != repeat or type(result.get('success')) is not bool:
                raise ValueError('Invalid completed result')
            if result.get('termination') not in ('task_success', 'official_action_limit'):
                raise ValueError('Incomplete termination marked completed')
            if result['success'] != (result['termination'] == 'task_success'):
                raise ValueError('Inconsistent terminal success')
        elif result.get('status') != 'failed' or result.get('exception') != 'UnStableError':
            raise ValueError('Unaudited infrastructure failure')
        terminal[name] = result
    return dict(planned_count=len(planned), terminal=terminal,
                interrupted=interrupted, pending=pending,
                interrupted_seeds=sorted({planned[n][0] for n in interrupted}),
                ready_for_final_analysis=not pending)


def require_complete(root):
    result = inventory(root)
    if not result['ready_for_final_analysis']:
        raise RuntimeError(f"{len(result['pending'])} planned attempts remain; no final statistics")
    return result

"""Explicit terminal/unknown/pending inventory for the fixed 384 attempts."""
import hashlib
import json
from pathlib import Path

CONDITIONS=('zero','true','mismatch','contact_state')


def planned_attempts():
    return {f'r{r}_s{steps}_{seed}_{arm}': (seed,r,steps,arm)
            for seed in range(100103,100119) for r in range(3)
            for steps in (720,2880) for arm in CONDITIONS}


def inventory(root):
    root=Path(root);planned=planned_attempts();interrupted={}
    for path in (root/'data/reports').glob('cwmj_m8_confirmation_interruption_*.json'):
        record=json.loads(path.read_text());name=record['interrupted_attempt']
        if name not in planned or name in interrupted:raise ValueError('unknown/duplicate interruption')
        hashes=record['original_files_sha256']
        if not hashes:raise ValueError('interruption evidence required')
        expected=(root/f'data/eval/cwmj_m8_confirm_{name}').resolve()
        for relative,digest in hashes.items():
            source=(root/relative).resolve()
            if not source.is_relative_to(expected):raise ValueError('evidence outside interrupted attempt')
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('evidence changed')
        interrupted[name]=record
    terminal={};pending=[]
    for name,(seed,repeat,steps,arm) in planned.items():
        path=root/f'data/eval/cwmj_m8_confirm_{name}/result.json'
        if name in interrupted:
            if path.exists():raise ValueError('interruption must not have fabricated terminal result')
            continue
        if not path.exists():pending.append(name);continue
        result=json.loads(path.read_text())
        if result.get('seed')!=seed or result.get('arm')!=arm:raise ValueError('attempt identity mismatch')
        if result.get('status')=='completed':
            if (result.get('repeat_index')!=repeat or result.get('training_steps')!=steps
                    or type(result.get('success')) is not bool):raise ValueError('terminal identity/success invalid')
            if result.get('termination') not in ('task_success','official_action_limit'):
                raise ValueError('truncation needs explicit audit')
            if result['success']!=(result['termination']=='task_success'):raise ValueError('inconsistent success')
        elif result.get('status')!='failed' or result.get('exception')!='UnStableError':
            raise ValueError('infrastructure failure requires inspection')
        terminal[name]=result
    return dict(planned_count=len(planned),terminal=terminal,interrupted=interrupted,pending=pending,
                interrupted_seeds=sorted({planned[n][0] for n in interrupted}),ready_for_final_analysis=not pending)

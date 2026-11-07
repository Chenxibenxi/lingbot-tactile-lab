"""Local serial experiment DAG; GPU commands must use their own lease launcher.

No service lifecycle commands, shell interpretation, or automatic retry of an
uncertain experiment. Completed stages are resumed only after artifact hashes
and the immutable plan match. Run under a detached process supervisor.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def run(plan_path, state_dir):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    stages = plan['stages']
    seen = set()
    for stage in stages:
        name = stage['id']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name) or name in seen:
            raise ValueError('invalid/duplicate stage id')
        if not set(stage.get('depends_on', [])).issubset(seen):
            raise ValueError('dependency must precede stage')
        if not stage['argv'] or not all(isinstance(x, str) for x in stage['argv']):
            raise ValueError('argv strings required')
        if not stage['outputs'] or any(not Path(p).is_absolute() for p in stage['outputs']):
            raise ValueError('absolute output paths required')
        if stage.get('gpu') and stage.get('uses_own_lease_launcher') is not True:
            raise ValueError('GPU stage requires own lease launcher')
        seen.add(name)
    root = Path(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digest = sha(plan_path)
        pin = root/'plan.sha256'
        if pin.exists():
            if pin.read_text() != digest:
                raise ValueError('plan changed; use a new queue')
        else:
            pin.write_text(digest)
        for stage in stages:
            name = stage['id']
            state_path = root/f'{name}.json'
            if state_path.exists():
                old = json.loads(state_path.read_text())
                if old['status'] != 'completed':
                    raise RuntimeError(f'{name}: prior nonterminal/failed attempt needs inspection, no blind retry')
                if any(not Path(p).is_file() or sha(p) != h for p, h in old['artifacts'].items()):
                    raise ValueError(f'{name}: completed artifacts changed')
                continue
            if any(Path(p).exists() for p in stage['outputs']):
                raise ValueError(f'{name}: unowned output exists')
            record = {'status': 'running', 'stage': name, 'plan_sha256': digest,
                      'scheduler_pid': os.getpid(), 'started_unix': time.time()}
            save(state_path, record)
            try:
                with (root/f'{name}.log').open('x') as log:
                    child = subprocess.Popen(stage['argv'], cwd=plan['cwd'], stdout=log,
                                             stderr=subprocess.STDOUT)
                    record['child_pid'] = child.pid
                    save(state_path, record)
                    code = child.wait()
                if code:
                    raise RuntimeError(f'exit code {code}')
                record['artifacts'] = {p: sha(p) for p in stage['outputs']}
                record['status'] = 'completed'
            except BaseException as exc:
                record.update(status='needs_inspection', error=f'{type(exc).__name__}: {exc}')
                save(state_path, record)
                raise
            record['finished_unix'] = time.time()
            save(state_path, record)
        save(root/'complete.json', {'status': 'completed', 'plan_sha256': digest,
                                   'stage_ids': [s['id'] for s in stages]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--state-dir', required=True)
    args = parser.parse_args()
    run(args.plan, args.state_dir)

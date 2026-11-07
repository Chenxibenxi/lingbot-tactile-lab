"""Observe the existing training scheduler, then launch exactly one eval batch."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path('/mnt/relaydata/lingbot-tactile-lab')
ROOT=R/'data/reports/cwmj_m8'
PY='/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def alive_identity(pid):
    path=Path('/proc')/str(pid)
    try:
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'src/research_queue.py' not in cmd or 'cwmj_m8/training_queue_plan.json' not in cmd:return None
        return stat[19]  # field22 starttime, accounting for pid/comm prefix
    except FileNotFoundError:return None


def run():
    with (ROOT/'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        process=json.loads((ROOT/'training_queue_process.json').read_text())
        pid=process['pid'];identity=alive_identity(pid)
        state=ROOT/'training_queue_state';completion=state/'complete.json'
        if identity is None and not completion.exists():raise RuntimeError('training scheduler absent; no restart')
        while not completion.exists():
            if alive_identity(pid)!=identity:raise RuntimeError('training scheduler ended without completion')
            for path in state.glob('train_*.json'):
                if json.loads(path.read_text())['status']=='needs_inspection':raise RuntimeError('training failed: '+path.name)
            time.sleep(30)
        plan_path=ROOT/'training_queue_plan.json';plan=json.loads(plan_path.read_text())
        done=json.loads(completion.read_text())
        if done['plan_sha256']!=sha(plan_path) or done['stage_ids']!=[s['id'] for s in plan['stages']]:raise ValueError('training completion matrix mismatch')
        for stage in plan['stages']:
            item=json.loads((state/(stage['id']+'.json')).read_text())
            if item['status']!='completed' or set(item['artifacts'])!=set(stage['outputs']):raise ValueError('missing completed training stage')
            for p,digest in item['artifacts'].items():
                if sha(Path(p))!=digest:raise ValueError('training artifact changed')
        code=json.loads((ROOT/'evaluation_code_lock.json').read_text())
        for p,digest in code['files'].items():
            if sha(R/p)!=digest:raise ValueError('evaluation source changed since preflight: '+p)
        logpath=ROOT/'confirmation_queue.log'
        if logpath.exists():raise RuntimeError('prior evaluation attempt requires inspection, no blind restart')
        with logpath.open('x') as log:
            child=subprocess.Popen([PY,'-u','-B',str(R/'work/run_cwmj_m8_confirmation_batch.py')],cwd=R,
                                   env=dict(os.environ,PYTHONPATH=str(R/'src')),stdout=log,stderr=subprocess.STDOUT)
            (ROOT/'confirmation_queue_process.json').write_text(json.dumps(dict(pid=child.pid,parent_pid=os.getpid()),indent=2))
            rc=child.wait()
        (ROOT/'continuation_result.json').write_text(json.dumps(dict(returncode=rc,status='completed' if rc==0 else 'needs_inspection'),indent=2))
        if rc:raise RuntimeError('evaluation batch failed; see preserved log')


if __name__=='__main__':run()

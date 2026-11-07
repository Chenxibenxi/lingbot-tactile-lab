"""Wait for the existing M8 supervisor; never restart it or alter GPU services."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

R=Path('/mnt/relaydata/lingbot-tactile-lab')
ROOT=R/'data/reports/recovery_development_v1'
M8=R/'data/reports/cwmj_m8'
PY='/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python'


def identity(pid):
    try:
        p=Path('/proc')/str(pid)
        cmd=(p/'cmdline').read_bytes()
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z' or b'work/continue_m8_after_training.py' not in cmd:return None
        return stat[19]
    except FileNotFoundError:return None


def run():
    with (ROOT/'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=M8/'continuation_result.json'
        pid=json.loads((M8/'continuation_process.json').read_text())['pid'];original=identity(pid)
        if original is None and not result.exists():raise RuntimeError('M8 supervisor absent; no restart')
        while not result.exists():
            if identity(pid)!=original:raise RuntimeError('M8 ended without completion result; inspect logs')
            time.sleep(30)
        if json.loads(result.read_text()).get('status')!='completed':raise RuntimeError('M8 requires inspection')
        if not (R/'data/reports/cwmj_m8_confirmation_summary/summary.json').exists():raise RuntimeError('M8 final report missing')
        manifest=json.loads((ROOT/'manifest.json').read_text())
        if manifest.get('gpu_execution_ready') is not True:raise ValueError('recovery implementation not frozen')
        for name,digest in manifest['files'].items():
            if hashlib.sha256((R/name).read_bytes()).hexdigest()!=digest:raise ValueError('recovery source drift: '+name)
        logpath=ROOT/'queue.log'
        if logpath.exists():raise RuntimeError('prior queue attempt exists; no blind restart')
        with logpath.open('x') as log:
            child=subprocess.Popen([PY,'-u','-B',str(R/'work/run_recovery_development_batch.py')],cwd=R,
                                   env=dict(os.environ,PYTHONPATH=str(R/'src')),stdout=log,stderr=subprocess.STDOUT)
            (ROOT/'queue_process.json').write_text(json.dumps(dict(pid=child.pid,parent_pid=os.getpid()),indent=2))
            code=child.wait()
        (ROOT/'queue_result.json').write_text(json.dumps(dict(returncode=code,status='completed' if code==0 else 'needs_inspection'),indent=2))
        if code:raise RuntimeError('recovery queue requires inspection')


if __name__=='__main__':run()

"""Resource-aware queue journal for newly registered experiments only."""
import fcntl
import json
import os
from pathlib import Path
from resource_aware_dispatch import dispatch
from dispatch_resume_contract import remaining
from gpu_resource_snapshot import snapshot


def write_new(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def run(root,matrix,*,launch,owner,resource_snapshot=snapshot):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    matrix=list(matrix)
    if any(not isinstance(x,str) or not x or '/' in x or x in ('.','..') for x in matrix):
        raise ValueError('simple unique trial names required')
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=root/'matrix.json'
        if plan.exists():
            if json.loads(plan.read_text())!=matrix:raise ValueError('registered matrix changed')
        else:write_new(plan,matrix)
        started={p.stem for p in (root/'started').glob('*.json')}
        receipts={p.stem:json.loads(p.read_text()) for p in (root/'receipts').glob('*.json')}
        pending=remaining(matrix,receipts,started)
        for name in ('started','receipts','events'):(root/name).mkdir(exist_ok=True)
        def execute(item):
            write_new(root/'started'/f'{item}.json',dict(pid=os.getpid(),item=item))
            try:result=launch(item)
            except Exception as exc:
                result=dict(returncode=-1,status='failed',resource_censored=None,error=type(exc).__name__+': '+str(exc))
            write_new(root/'receipts'/f'{item}.json',result)
            return result
        def save(value):
            index=len(list((root/'events').glob('*.json')))
            write_new(root/'events'/f'{index:06d}.json',value)
        return dispatch(pending,snapshot=resource_snapshot,launch=execute,save=save,owner=owner)

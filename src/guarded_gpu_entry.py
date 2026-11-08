"""New-protocol entry wrapper; preserves PID ancestry and original runner checks."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from gpu_resource_snapshot import snapshot
from gpu_admission_diagnostic import reasons


def check(state,owner):
    problems=reasons(state.get('lease_status',{}),owner=owner,
                     ancestor_pids={x['ppid'] for x in state.get('ancestors',[])[:2]})
    if state.get('probe_ok') is not True:problems.append('resource_probe_failed')
    if state.get('compute_pids'):problems.append('gpu_compute_active')
    return problems


def enter(script,args,evidence,*,owner,read_snapshot=snapshot,execute=runpy.run_path):
    script=Path(script).resolve(strict=True)
    state=read_snapshot();issues=check(state,owner)
    # Exclusive write before entering user/model code; failure to record is fatal.
    with Path(evidence).open('x') as f:
        json.dump(dict(script=str(script),pid=os.getpid(),snapshot=state,reasons=issues,admitted=not issues),f,indent=2)
        f.flush();os.fsync(f.fileno())
    if issues:return 75
    old=sys.argv
    try:
        sys.argv=[str(script),*args]
        execute(str(script),run_name='__main__')
    finally:sys.argv=old
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--owner',required=True)
    p.add_argument('script');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    raise SystemExit(enter(a.script,a.args,a.evidence,owner=a.owner))

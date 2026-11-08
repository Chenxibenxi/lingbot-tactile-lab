"""Read-only, bounded resource snapshot; never controls Strata or leases."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
LEASE_TOOL='/home/admin001/ai-video-studio/tools/gpu_lease.py'

def snapshot(run=subprocess.run):
    result=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),probe_ok=False,
                compute_pids=[],lease_status={},errors=[])
    for kind,args in [('lease_status',[sys.executable,LEASE_TOOL,'status','--json']),
                      ('compute_pids',['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])]:
        try:
            p=run(args,capture_output=True,text=True,check=True,timeout=10)
            if kind=='lease_status':
                value=json.loads(p.stdout)
                if not isinstance(value,dict) or value.get('state') not in ('free','held','stale') or not isinstance(value.get('waiting'),list):
                    raise ValueError('invalid lease status schema')
            else:value=[int(x.strip()) for x in p.stdout.splitlines() if x.strip()]
            result[kind]=value
        except Exception as exc:result['errors'].append(dict(probe=kind,error=type(exc).__name__+': '+str(exc)))
    parents=[];pid=os.getpid()
    try:
        for _ in range(8):
            status=Path(f'/proc/{pid}/status').read_text()
            parent=int(next(x.split(':',1)[1] for x in status.splitlines() if x.startswith('PPid:')))
            parents.append(dict(pid=pid,ppid=parent))
            if parent<=1:break
            pid=parent
    except (OSError,ValueError,StopIteration) as exc:
        result['errors'].append(dict(probe='ancestors',error=str(exc)))
    result['ancestors']=parents;result['probe_ok']=not result['errors'];return result

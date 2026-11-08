"""Three finite independent benchmark processes; one final report, no retries."""
import json
import os
from pathlib import Path
import subprocess
from verify_microbatch_benchmark_v2 import R,verify
from report_microbatch_benchmark_v2 import report


def write(path,data):
    with Path(path).open('x') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())


def run():
    digest=verify();root=R/'data/reports/microbatch_benchmark_v2'
    write(root/'batch_started.json',dict(pid=os.getpid(),manifest_sha256=digest))
    attempts=[]
    for micro in (1,2,4):
        verify()
        path=root/f'micro{micro}'
        if path.exists():raise ValueError('no retry/overwrite of an existing attempt')
        compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
        if compute.strip():
            write(root/'queue_result.json',dict(status='paused',reason='GPU occupied',attempts=attempts));return
        with (root/f'micro{micro}.log').open('x') as log:
            completed=subprocess.run(['bash',str(R/'scripts/run_microbatch_benchmark_v2.sh'),str(micro)],stdout=log,stderr=subprocess.STDOUT)
        result=path/'result.json'
        if not result.exists():
            path.mkdir(exist_ok=True)
            write(result,dict(status='failed',microbatch=micro,error=f'launcher exit {completed.returncode}; no terminal result'))
        value=json.loads(result.read_text())
        attempts.append(dict(microbatch=micro,status=value['status'],returncode=completed.returncode))
    out=report(R)
    write(root/'queue_result.json',dict(status='completed',attempts=attempts,report=str(out),training_approval=False))

if __name__=='__main__':run()

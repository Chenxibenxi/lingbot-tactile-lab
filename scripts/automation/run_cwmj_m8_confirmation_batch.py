"""Serial unattended evaluations; no rerun of ambiguous partial attempts."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

R=Path('/mnt/relaydata/lingbot-tactile-lab')
PY='/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python'
LEASE='/home/admin001/ai-video-studio/tools/gpu_lease.py'
sys.path.insert(0,str(R/'src'))
from m8_completion_inventory import planned_attempts
from verify_cwmj_m8_confirmation import verify_confirmation


def run():
    root=R/'data/reports/cwmj_m8_confirmation';root.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(R/'src'))
    def call(args):subprocess.run(args,cwd=R,env=env,check=True)
    with (root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (R/'data/reports/cwmj_m8/confirmation_manifest.json').exists():
            call([PY,'-B','src/prepare_cwmj_m8_confirmation.py'])
        for name,(seed,repeat,steps,arm) in planned_attempts().items():
            verify_confirmation(R,seed,repeat,arm,steps)
            trial=R/f'data/eval/cwmj_m8_confirm_{name}'
            result=trial/'result.json'
            if trial.exists() and not result.exists():raise RuntimeError('partial/live attempt; inspection required: '+name)
            if not result.exists():
                logpath=root/f'{name}.log'
                if logpath.exists():raise RuntimeError('previous launcher requires inspection: '+name)
                while True:
                    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
                    lease=json.loads(subprocess.run(['python3',LEASE,'status','--json'],check=True,capture_output=True,text=True).stdout)
                    if not gpu and lease['state'] in ('free','stale'):break
                    time.sleep(30)
                if shutil.disk_usage(R).free<3*1024**3:raise RuntimeError('disk headroom insufficient')
                with logpath.open('x') as log:
                    child=subprocess.run(['bash','work/run_cwmj_m8_confirmation.sh',str(seed),arm,str(repeat),str(steps)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
                if not result.exists():
                    (root/f'{name}.needs_inspection.json').write_text(json.dumps(dict(name=name,returncode=child.returncode,status='unknown_not_scored')))
                    raise RuntimeError('missing terminal result; preserve partial attempt')
            row=json.loads(result.read_text())
            if not (root/name/'report.md').exists():
                call([PY,'-B','src/report_cwmj_m8_confirmation.py','--seed',str(seed),'--repeat',str(repeat),'--arm',arm,'--steps',str(steps)])
            call([PY,'-B','src/queue_strata_m8_analysis.py'])
            if row.get('status')!='completed' and row.get('exception')!='UnStableError':raise RuntimeError('infrastructure failure needs inspection')
            if row.get('status')=='completed' and row.get('termination') not in ('task_success','official_action_limit'):raise RuntimeError('truncated episode needs inspection')
            print('EPISODE_REPORTED',name,flush=True)
        call([PY,'-B','src/report_cwmj_m8_confirmation_batch.py'])
        call([PY,'-B','src/queue_strata_m8_analysis.py','--execute'])


if __name__=='__main__':run()

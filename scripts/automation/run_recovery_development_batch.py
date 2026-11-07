"""Eight fixed development attempts after M8, with no blind retries."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

R=Path('/mnt/relaydata/lingbot-tactile-lab')
PY='/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python'
LEASE='/home/admin001/ai-video-studio/tools/gpu_lease.py'


def run():
    prerequisite=R/'data/reports/cwmj_m8/continuation_result.json'
    if not prerequisite.exists() or json.loads(prerequisite.read_text()).get('status')!='completed':
        raise RuntimeError('M8 training/evaluation must finish before recovery GPU work')
    root=R/'data/reports/recovery_development';root.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(R/'src'))
    with (root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for seed in range(100119,100123):
            for condition in ('expert_control','left_open_once_recovery'):
                name=f'recovery_dev_{seed}_{condition}';trial=R/'data/paired'/name;logpath=root/(name+'.log')
                result=trial/'result.json';failure=trial/'failure.json'
                if trial.exists() or logpath.exists():raise RuntimeError('existing attempt requires inspection; no automatic restart: '+name)
                while True:
                    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
                    lease=json.loads(subprocess.run(['python3',LEASE,'status','--json'],check=True,capture_output=True,text=True).stdout)
                    if not gpu and lease['state'] in ('free','stale'):break
                    time.sleep(30)
                if shutil.disk_usage(R).free<3*1024**3:raise RuntimeError('insufficient disk headroom')
                with logpath.open('x') as log:
                    completed=subprocess.run(['bash',str(R/'work/run_recovery_development.sh'),str(seed),condition],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
                if not result.exists() and not failure.exists():
                    # The launch process is terminal, so it is safe to record
                    # infrastructure failure without inventing a simulator result.
                    trial.mkdir(exist_ok=True)
                    failure.write_text(json.dumps(dict(status='needs_inspection',seed=seed,condition=condition,
                        error_type='LauncherWithoutTerminal',error=f'launcher exit {completed.returncode}; outcome unknown',training_approval=False),indent=2))
                subprocess.run([PY,'-B',str(R/'src/report_recovery_development.py'),'--trial',str(trial),'--out',str(root/name)],cwd=R,env=env,check=True)
                if failure.exists():
                    record=json.loads(failure.read_text())
                    if record.get('error_type')!='OnlineExpertPlanError':raise RuntimeError('infrastructure exception; inspect preserved report')
                print('RECOVERY_REPORT_READY',name,flush=True)
        subprocess.run([PY,'-B',str(R/'src/report_recovery_development_batch.py')],cwd=R,env=env,check=True)


if __name__=='__main__':run()

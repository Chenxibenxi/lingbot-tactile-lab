"""CPU-only reporting worker for completed fixed training tasks; no GPU actions."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

R=Path('/mnt/relaydata/lingbot-tactile-lab')
ROOT=R/'data/reports/cwmj_m8'
PY='/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    with (ROOT/'duration_reports.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        script=R/'src/report_m8_duration_curves.py';script_sha=digest(script)
        pid=json.loads((ROOT/'training_queue_process.json').read_text())['pid']
        state=ROOT/'training_queue_state';done=set()
        while len(done)<12:
            if digest(script)!=script_sha:raise RuntimeError('report implementation changed during worker run')
            for repeat in range(3):
                for arm in ('zero','true','mismatch','contact_state'):
                    name=f'train_r{repeat}_{arm}'
                    if name in done:continue
                    path=state/(name+'.json')
                    if not path.exists():continue
                    item=json.loads(path.read_text())
                    if item['status']=='needs_inspection':raise RuntimeError('training needs inspection: '+name)
                    if item['status']!='completed':continue
                    for p,h in item['artifacts'].items():
                        if digest(Path(p))!=h:raise ValueError('completed training artifact changed')
                    source=R/f'data/reports/cwmj_m8_training_r{repeat}_{arm}/result.json'
                    output=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m8-training')/f'duration_r{repeat}_{arm}'
                    if output.exists():
                        summary=json.loads((output/'summary.json').read_text())
                        if summary['source_sha256']!=digest(source) or not (output/'report.md').exists() or not (output/'duration_curve.png').exists():
                            raise ValueError('existing duration report incomplete or stale')
                    else:
                        subprocess.run([PY,'-B',str(script),'--repeat',str(repeat),'--arm',arm],cwd=R,check=True)
                    done.add(name)
                    print('DURATION_REPORT_READY',name,flush=True)
            snapshot=dict(completed_reports=sorted(done),reporter_sha256=script_sha)
            temp=ROOT/'duration_reports_status.tmp';temp.write_text(json.dumps(snapshot,indent=2));temp.replace(ROOT/'duration_reports_status.json')
            if len(done)==12:break
            if (state/'complete.json').exists():raise RuntimeError('training claims completion but report inputs incomplete')
            cmd=Path('/proc')/str(pid)/'cmdline'
            if not cmd.exists() or b'research_queue.py' not in cmd.read_bytes() or b'cwmj_m8/training_queue_plan.json' not in cmd.read_bytes():
                raise RuntimeError('training scheduler absent; no restart')
            time.sleep(30)


if __name__=='__main__':run()

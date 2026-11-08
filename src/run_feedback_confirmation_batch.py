"""Finite 384-attempt queue, no retries, exclusive start and final source audit."""
import json
import os
import shutil
import subprocess
from pathlib import Path
from verify_feedback_confirmation import R,SEEDS,ARMS,identity,verify
from validate_expert_data import sha256
from report_feedback_confirmation_episode import report as episode_report
from report_feedback_confirmation_final import report as final_report


def matrix():
    return [(s,r,c,a) for s in SEEDS for r in (0,1,2) for c in (10,1) for a in ARMS]


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def run():
    verify(SEEDS[0],ARMS[0],0,10)
    root=R/'data/reports/feedback_confirmation_v1'
    write(root/'batch_started.json',dict(pid=os.getpid(),manifest_sha256=sha256(root/'manifest.json')))
    attempts=[]
    for seed,repeat,cadence,arm in matrix():
        verify(seed,arm,repeat,cadence)
        name=identity(seed,arm,repeat,cadence);trial=R/'data/eval/feedback_confirmation_v1'/name
        if trial.exists():raise ValueError('existing attempt; inspect without automatic retry')
        compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
        if compute.strip() or shutil.disk_usage(root).free<2*1024**3:
            write(root/'queue_result.json',dict(status='paused',reason='GPU occupied or report disk low',attempts=attempts));return
        with (root/(name+'.log')).open('x') as log:
            process=subprocess.run(['bash',str(R/'scripts/run_feedback_confirmation.sh'),str(seed),str(repeat),arm,str(cadence)],stdout=log,stderr=subprocess.STDOUT)
        result_path=trial/'result.json'
        if not result_path.exists():
            trial.mkdir(exist_ok=True)
            write(result_path,dict(status='failed',seed=seed,repeat_index=repeat,replan_steps=cadence,arm=arm,
                error=f'launcher exit {process.returncode}; no terminal result',requests=[],training_approval=False))
        result=json.loads(result_path.read_text())
        receipt=dict(seed=seed,repeat=repeat,cadence=cadence,arm=arm,status=result['status'],
                     success=result.get('success'),returncode=process.returncode,result_sha256=sha256(result_path))
        attempts.append(receipt);write(root/(name+'_receipt.json'),receipt)
        try:episode_report(trial,root/'reports'/name)
        except Exception as exc:
            write(root/'queue_result.json',dict(status='paused',reason='episode report error: '+str(exc),attempts=attempts));raise
    # Terminal execution state is distinct from successful source audit or report generation.
    write(root/'queue_result.json',dict(status='completed',attempts=attempts,training_approval=False,
        final_audit_required=True))
    summary=final_report(R)
    write(root/'final_report_receipt.json',dict(report_complete=True,technical_gate=summary['technical_gate'],
        report_sha256=sha256(R/'data/reports/feedback_confirmation_v1_final/report.md'),training_approval=False))

if __name__=='__main__':run()

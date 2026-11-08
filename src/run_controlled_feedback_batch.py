"""Finite registered pilot batch; no polling, retries, or Strata lifecycle actions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from verify_controlled_feedback_pilot import R,verify
from report_controlled_feedback_pilot import report


def write_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())


def run():
    root=R/'data/reports/controlled_feedback_pilot_v1'
    verify(100127,'zero',10)
    write_new(root/'batch_started.json',dict(pid=os.getpid()))
    attempts=[]
    for seed in (100127,100128):
        for cadence in (10,1):
            for arm in ('zero','true','mismatch','contact_state'):
                verify(seed,arm,cadence)
                name=f's{seed}_{arm}_c{cadence}'
                trial=R/'data/eval/controlled_feedback_pilot_v1'/name
                if trial.exists():raise ValueError('attempt already started; inspect rather than retry')
                compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
                if compute.strip() or shutil.disk_usage(root).free<2*1024**3:
                    write_new(root/'queue_result.json',dict(status='paused',reason='GPU busy or report disk low',attempts=attempts));return
                with (root/(name+'.log')).open('x') as log:
                    completed=subprocess.run(['bash',str(R/'scripts/run_controlled_feedback_pilot.sh'),str(seed),arm,str(cadence)],stdout=log,stderr=subprocess.STDOUT)
                result_path=trial/'result.json'
                if not result_path.exists():
                    trial.mkdir(exist_ok=True)
                    write_new(result_path,dict(status='failed',seed=seed,arm=arm,replan_steps=cadence,
                        exception='LauncherFailure',error=f'returncode={completed.returncode}; no terminal result',requests=[],training_approval=False))
                result=json.loads(result_path.read_text())
                report(trial,root/'reports'/name)
                receipt=dict(name=name,seed=seed,cadence=cadence,arm=arm,status=result['status'],
                    success=result.get('success'),resource_censored=result.get('resource_censored'),returncode=completed.returncode)
                attempts.append(receipt);write_new(root/(name+'_receipt.json'),receipt)
    write_new(root/'queue_result.json',dict(status='completed',attempts=attempts,
        confirmation_eligible=False,training_approval=False,independent_final_audit_required=True))


if __name__=='__main__':run()

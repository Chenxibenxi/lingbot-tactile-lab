"""Resource-aware registered v3 queue; no automatic retry of started attempts."""
import json,subprocess,shutil
from pathlib import Path
from durable_resource_queue import run as dispatch, write_new
from gpu_resource_snapshot import snapshot
from verify_feedback_confirmation_v3 import R,SEEDS,ARMS,identity,verify
from report_feedback_confirmation_episode_v3 import report
from report_feedback_confirmation_final_v3 import report as final_report

def run():
    verify(SEEDS[0],ARMS[0],0,10)
    root=R/'data/reports/feedback_confirmation_v3'
    matrix={identity(s,a,r,c):(s,r,c,a) for s in SEEDS for r in (0,1,2) for c in (10,1) for a in ARMS}
    def launch(name):
        s,r,c,a=matrix[name];verify(s,a,r,c)
        trial=R/'data/eval/feedback_confirmation_v3'/name
        if trial.exists():raise ValueError('existing trial cannot be retried')
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.run(['bash',str(R/'scripts/run_feedback_confirmation_v3.sh'),str(s),str(r),a,str(c)],stdout=log,stderr=subprocess.STDOUT)
        path=trial/'result.json'
        if not path.exists():
            trial.mkdir(exist_ok=True)
            with path.open('x') as f:json.dump(dict(status='failed',seed=s,repeat_index=r,replan_steps=c,arm=a,requests=[],training_approval=False,error='launcher failure before terminal result'),f)
        result=json.loads(path.read_text());report(trial,root/'reports'/name)
        return dict(seed=s,repeat=r,cadence=c,arm=a,status=result['status'],returncode=proc.returncode,
                    resource_censored=result.get('resource_censored'),success=result.get('success'))
    def resources():
        state=snapshot()
        if shutil.disk_usage(root).free<2*1024**3:
            state['probe_ok']=False;state.setdefault('errors',[]).append({'probe':'report_disk','error':'less than 2 GiB free'})
        return state
    result=dispatch(root/'journal',list(matrix),launch=launch,owner='lingbot-tactile-lab/real-v2-bf16-stream',resource_snapshot=resources)
    if result['status']=='completed':
        receipts=[json.loads((root/'journal/receipts'/f'{name}.json').read_text()) for name in matrix]
        target=root/'queue_result.json'
        if not target.exists():
            with target.open('x') as f:json.dump(dict(status='completed',attempts=receipts,training_approval=False),f,indent=2)
        final_dir=R/'data/reports/feedback_confirmation_v3_final'
        receipt=root/'final_report_receipt.json'
        if not receipt.exists():
            if final_dir.exists():raise RuntimeError('partial final report requires inspection; do not overwrite')
            summary=final_report(R)
            write_new(receipt,dict(report_complete=True,technical_gate=summary['technical_gate'],training_approval=False))
    return result
if __name__=='__main__':run()

"""Remaining seven fixed development diagnostics, resource-aware and no retries."""
import json,subprocess
from pathlib import Path
from durable_resource_queue import run,write_new
from gpu_resource_snapshot import snapshot
from audit_feedback_prediction_trace import audit
from report_controlled_feedback_pilot_v2 import report
from verify_feedback_motion_diagnostic import verify
R=Path(__file__).resolve().parents[1];B=R/'data/reports/feedback_motion_diagnostic_run_v1'
def main():
 matrix={f's{s}_{a}_c{c}':(s,a,c) for s in (100145,100146) for a in ('zero','true') for c in (10,1) if (s,a,c)!=(100145,'zero',10)}
 def launch(name):
  s,a,c=matrix[name];verify(s,a,c);trial=R/'data/eval/feedback_motion_diagnostic_v1'/name
  if trial.exists():raise ValueError('existing trial; no retry')
  with (B/(name+'.log')).open('x') as log:p=subprocess.run(['bash',str(R/'scripts/run_feedback_motion_diagnostic.sh'),str(s),a,str(c)],stdout=log,stderr=subprocess.STDOUT)
  f=trial/'result.json'
  if not f.exists():return dict(returncode=p.returncode or -1,status='failed',resource_censored=None,error='no terminal result')
  v=json.loads(f.read_text());out=B/'reports'/name;report(trial,out)
  if p.returncode==0 and v['status']=='completed' and v.get('resource_censored') is False:
   evidence=audit(trial);write_new(out/'trace_audit.json',evidence)
   with (out/'report.md').open('a') as f:f.write('\n\n本场为开发集被动诊断，额外保存完整预测块和实际关节状态；trace_audit.json核对预测前缀、执行指令与物理时钟。不是新增独立确认样本。\n')
  return dict(returncode=p.returncode,status=v['status'],resource_censored=v.get('resource_censored'),success=v.get('success'))
 result=run(B/'journal_remaining',list(matrix),launch=launch,owner='lingbot-tactile-lab/real-v2-bf16-stream',resource_snapshot=snapshot)
 print(json.dumps(result))
if __name__=='__main__':main()

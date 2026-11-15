from pathlib import Path
import os,json,time,subprocess,select,hashlib,fcntl
R=Path('/mnt/relaydata/lingbot-tactile-lab');W=R/'work/cwmj_epoch_boundary_finish_v2'
lock=(W/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def dump(**x):
 p=W/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 identity=json.loads((W/'source_process.json').read_text());p=Path('/proc')/str(identity['pid'])
 if p.exists():
  assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==identity['start_ticks']
  fd=os.pidfd_open(identity['pid']);dump(status='waiting_existing_training_exit',source=identity);select.select([fd],[],[]);os.close(fd)
 s=json.loads((R/'data/reports/cwmj_fixed_dual_until_20261115_1900/status.json').read_text())
 assert s['status']=='completed' and s.get('stop_reason')=='authorized_deadline',s
 init=json.loads((R/'data/reports/cwmj_fixed_dual_until_20261115_1900/initialization.json').read_text());n=init['steps_per_epoch'];remainder=s['step']%n
 source=R/'data/cwmj_fixed_dual_until_20261115_1900/deadline_final.pt';assert source.is_file()
 if remainder:
  dump(status='finishing_partial_epoch',source_status=s,remaining_updates=n-remainder)
  subprocess.run(['bash',str(W/'run.sh')],cwd=R,check=True)
  final=json.loads((R/'data/reports/cwmj_epoch_boundary_finish_v2/status.json').read_text());assert final['status']=='completed' and final['step']%n==0
  checkpoint=R/'data/cwmj_epoch_boundary_finish_v2/epoch_final.pt'
 else:checkpoint=source;final=s
 assert final['step']==final['completed_epochs']*n
 result={'checkpoint':str(checkpoint),'sha256':digest(checkpoint),'step':final['step'],'completed_epochs':final['completed_epochs'],'evaluation_plan':str(W/'evaluation_plan.json'),'training_finished':True,'evaluation_started':False}
 (W/'evaluation_ready.json').write_text(json.dumps(result,indent=2));dump(status='ready_for_evaluation',**result)
except BaseException as e:
 dump(status='failed',error=repr(e));raise

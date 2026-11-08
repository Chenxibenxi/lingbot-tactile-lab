import json,tempfile,unittest
from pathlib import Path
from guarded_gpu_entry import enter
class Entry(unittest.TestCase):
 def test_denied_entry_records_reason_and_never_executes(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);script=p/'runner.py';script.write_text('')
   calls=[];state=dict(probe_ok=True,compute_pids=[],ancestors=[{'ppid':12}],lease_status=dict(state='held',holder='ours',pid_alive=True,lease={'pid':12},remaining_s=2000,waiting=[{'owner':'secretary'}]))
   self.assertEqual(enter(script,[],p/'e.json',owner='ours',read_snapshot=lambda:state,execute=lambda *a,**k:calls.append(a)),75)
   self.assertEqual(calls,[]);self.assertEqual(json.loads((p/'e.json').read_text())['reasons'],['other_gpu_waiter'])
 def test_admitted_entry_preserves_pid_and_argument_contract(self):
  import os,sys
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);script=p/'runner.py';script.write_text('');pid=os.getpid();old=sys.argv
   state=dict(probe_ok=True,compute_pids=[],ancestors=[{'ppid':12}],lease_status=dict(state='held',holder='ours',pid_alive=True,lease={'pid':12},remaining_s=2000,waiting=[]))
   def execute(path,run_name):
    self.assertEqual(os.getpid(),pid);self.assertEqual(sys.argv,[str(script),'--seed','1']);self.assertEqual(run_name,'__main__')
   self.assertEqual(enter(script,['--seed','1'],p/'e.json',owner='ours',read_snapshot=lambda:state,execute=execute),0)
   self.assertIs(sys.argv,old)

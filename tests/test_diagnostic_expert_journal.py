import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from diagnostic_expert_journal import ExpertJournal
class ExpertJournalContract(unittest.TestCase):
 def test_observer_preserves_return_and_exception_and_restores_functions(self):
  marker=object();fault=ValueError('planner failed');robot=SimpleNamespace(get_left_arm_real_jointState=lambda:[0]*7,get_right_arm_real_jointState=lambda:[0]*7)
  task=SimpleNamespace(robot=robot,_force_probe=SimpleNamespace(step=0),eval_success=False)
  calls=[]
  def act(command,*args,**kwargs):calls.append((command,args,kwargs));task._force_probe.step+=5;return marker
  task.take_action=act
  def plan(robot,action,arm):
   if arm=='right':raise fault
   return [[0]*6,[1]*6]
  paired=lambda task,stage:marker;api=SimpleNamespace(_plan=plan,_paired_actions=paired);action=SimpleNamespace(target_pose=[0]*7)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'journal.jsonl'
   with ExpertJournal(task,api,p):
    self.assertIs(api._paired_actions(task,'lift'),marker)
    self.assertEqual(api._plan(robot,action,'left'),[[0]*6,[1]*6])
    self.assertIs(task.take_action([1]*14,action_type='qpos'),marker)
    with self.assertRaises(ValueError) as caught:api._plan(robot,action,'right')
    self.assertIs(caught.exception,fault)
   self.assertIs(api._plan,plan);self.assertIs(api._paired_actions,paired);self.assertIs(task.take_action,act)
   rows=[json.loads(line) for line in p.read_text().splitlines()]
   failure=next(r for r in rows if r['event']=='plan_exception');self.assertEqual((failure['phase'],failure['action_index'],failure['physics_step']),('lift',1,5));self.assertEqual(len(calls),1)

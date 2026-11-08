import unittest
from gpu_admission_diagnostic import reasons
class Admission(unittest.TestCase):
    def test_each_refusal_has_specific_reason(self):
        base=dict(state='held',holder='ours',pid_alive=True,lease={'pid':12},remaining_s=2100,waiting=[])
        self.assertEqual(reasons(base,owner='ours',ancestor_pids={12,13}),[])
        for change,expected in [({'state':'free'},'lease_not_held'),({'holder':'other'},'owner_mismatch'),({'pid_alive':False},'holder_pid_not_alive'),({'lease':{'pid':99}},'holder_not_ancestor'),({'remaining_s':599},'insufficient_ttl'),({'waiting':[{'owner':'other'}]},'other_gpu_waiter')]:
            self.assertEqual(reasons(dict(base,**change),owner='ours',ancestor_pids={12,13}),[expected])
    def test_own_waiter_not_rejected(self):
        x=dict(state='held',holder='ours',pid_alive=True,lease={'pid':1},remaining_s=600,waiting=[{'owner':'ours'}])
        self.assertEqual(reasons(x,owner='ours',ancestor_pids={1}),[])

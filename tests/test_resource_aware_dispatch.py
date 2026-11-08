import unittest
from resource_aware_dispatch import dispatch
class Dispatch(unittest.TestCase):
    def run_case(self,snap,result):
        launches=[];saved=[]
        def launch(item):launches.append(item);return result
        answer=dispatch(range(384),snapshot=lambda:snap,launch=launch,save=saved.append,owner='ours')
        self.assertEqual(saved,[answer]);return launches,answer
    def free(self):return dict(probe_ok=True,compute_pids=[],lease_status={'state':'free','waiting':[]})
    def test_other_waiter_consumes_no_attempt(self):
        x=self.free();x['lease_status']['waiting']=[{'owner':'secretary'}]
        calls,r=self.run_case(x,{})
        self.assertEqual(calls,[]);self.assertEqual(r['pending_item'],0)
    def test_race_after_preflight_stops_after_first_rejection(self):
        calls,r=self.run_case(self.free(),dict(returncode=1,status='failed',error='lease refused'))
        self.assertEqual(calls,[0]);self.assertEqual(r['reason'],'attempt_technical_error')
    def test_ordinary_task_failure_is_not_infrastructure_failure(self):
        calls,r=self.run_case(self.free(),dict(returncode=0,status='completed',resource_censored=False,success=False))
        self.assertEqual(len(calls),384);self.assertEqual(r['status'],'completed')
    def test_probe_error_held_lease_and_compute_each_pause(self):
        for delta in ({'probe_ok':False},{'compute_pids':[42]}, {'lease_status':{'state':'held','waiting':[]}}):
            calls,_=self.run_case(dict(self.free(),**delta),{});self.assertEqual(calls,[])
    def test_censoring_stops_dispatch(self):
        calls,_=self.run_case(self.free(),dict(returncode=0,status='completed',resource_censored=True))
        self.assertEqual(calls,[0])

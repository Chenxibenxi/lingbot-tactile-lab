import copy
import unittest
from controlled_feedback_timing import request_timing,execution_timing
from audit_controlled_feedback_timing import audit_requests


class ExecutionAuditTest(unittest.TestCase):
    def build(self,cadence=10,count=12):
        requests=[];ids=[];before=[];after=[]
        for pos in range(0,count,cadence):
            req=request_timing(seed=100127,executed_actions=pos,cadence=cadence,
                               capture_step=pos*5,history_anchor=pos*5,window_steps=25,dt_s=.004)
            req.update(request_index=len(requests),physics_step=pos*5,executed_action_timing=[])
            for i in range(pos,min(pos+cadence,count)):
                ids.append(len(requests));before.append(i*5);after.append((i+1)*5)
                req['executed_action_timing'].append(execution_timing(req,physics_before=i*5,physics_after=(i+1)*5,inference_seconds=.6))
            requests.append(req)
        return requests,dict(seed=100127,cadence=cadence,official_success=True,
            action_request_indices=ids,action_before=before,action_after=after)

    def test_both_cadences_and_short_success_tail(self):
        for cadence in (1,10):
            requests,args=self.build(cadence)
            self.assertTrue(audit_requests(requests,**args)['actual_cadence_verified'])

    def test_reject_missing_action_wrong_noise_and_future_anchor(self):
        for field,value in (('noise_seed',1),('history_anchor',1),('executed_actions',1)):
            requests,args=self.build();requests[0][field]=value
            with self.assertRaises(ValueError):audit_requests(requests,**args)
        requests,args=self.build();args['action_request_indices'][0]=1
        with self.assertRaises(ValueError):audit_requests(requests,**args)

    def test_nonterminal_short_segment_not_accepted(self):
        requests,args=self.build();args['official_success']=False
        with self.assertRaises(ValueError):audit_requests(requests,**args)

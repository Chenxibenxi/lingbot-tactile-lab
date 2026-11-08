import copy
import unittest
from test_feedback_topp_diagnostics import Planner
from types import SimpleNamespace
import numpy as np
from feedback_topp_diagnostics import observe_topp
from audit_feedback_topp import audit
from feedback_outcome_contract_v2 import validate_outcome

class Audit(unittest.TestCase):
    def evidence(self,failure=False):
        events=[];robot=SimpleNamespace(left_mplib_planner=Planner(),right_mplib_planner=Planner())
        robot.right_mplib_planner.failure=failure
        for i in range(2):
            with observe_topp(robot,events.append,lambda:dict(action_index=i,request_index=0,physics_step=i*50)) as trace:
                for planner in (robot.left_mplib_planner,robot.right_mplib_planner):
                    try:planner.TOPP(np.zeros((2,6)),.004)
                    except ValueError:pass
                trace.require_valid()
        return events,[dict(request_index=0,executed_action_timing=[dict(physics_step_before=0),dict(physics_step_before=50)])]
    def test_roundtrip_and_visible_fallback(self):
        for failure in (False,True):
            events,requests=self.evidence(failure)
            result=audit(events,requests,2)
            self.assertEqual(result['fallback_calls'],2 if failure else 0)
    def test_missing_duplicate_and_wrong_context_rejected(self):
        events,requests=self.evidence()
        variants=[events[:-1],events+events[:1]]
        bad=copy.deepcopy(events);bad[4]['context']['action_index']=0;variants.append(bad)
        bad=copy.deepcopy(events);bad[1]['duration_seconds']=float('nan');variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):audit(bad,requests,2)
    def test_budget_and_unknown_contract(self):
        result=dict(status='completed',resource_censored=False,actions=400,physics_steps=150000,
                    original_action_limit=400,denoise_steps=10,training_steps=2880,temporal_ensemble=False,
                    training_approval=False,success=False,termination='official_action_limit')
        self.assertTrue(validate_outcome(result))
        for changed in (dict(physics_steps=200001),dict(resource_censored=True),dict(actions=399)):
            with self.assertRaises(ValueError):validate_outcome(dict(result,**changed))

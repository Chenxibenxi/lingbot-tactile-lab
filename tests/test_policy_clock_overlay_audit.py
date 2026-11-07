import json,tempfile,unittest
import tests.test_validate_policy_trace as fixtures
from validate_policy_trace import validate
class ExplicitClockRecovery(unittest.TestCase):
    def test_expert_clock_requires_opt_in_and_all_real_policy_steps(self):
        with tempfile.TemporaryDirectory() as td:
            raw,obs,policy=fixtures.PolicyTraceAuditTests().make_trace(td)
            rows=[json.loads(x) for x in raw.read_text().splitlines()]
            rows[0]['clock_origin']='first_instrumented_expert_step_after_setup_stability'
            for row in rows[1:]:row['motion_phase']='policy_action'
            raw.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaises(ValueError):validate(raw,obs,policy)
            self.assertEqual(validate(raw,obs,policy,allow_policy_calls_on_expert_overlay=True)['actual_infer_attempts'],2)
            rows[10]['motion_phase']='expert_dense'
            raw.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaises(ValueError):validate(raw,obs,policy,allow_policy_calls_on_expert_overlay=True)
if __name__=='__main__':unittest.main()

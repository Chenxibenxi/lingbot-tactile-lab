"""CPU-only causal joins at actual policy requests; never training approval."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS

from force_probe import ForceProbe
from tests.test_force_probe import fake_robot
from validate_policy_trace import validate


class PolicyTraceAuditTests(unittest.TestCase):
    def make_trace(self,root):
        robot=fake_robot();scene=NS(get_contacts=lambda:[])
        probe=ForceProbe(root,task='lift_pot',seed=100048,episode=0,dt=.004,robot=robot)
        frame={'observation':{'head_camera':{'rgb':NS(shape=(240,320,3))}}}
        probe.record_observation(frame,0);probe.record_policy_request()
        for _ in range(25):probe.sample(scene,robot,1)
        probe.record_observation(frame,1);probe.record_policy_request()
        raw=probe.raw_path;probe.close()
        return raw,raw.with_suffix('.obs.jsonl'),raw.with_suffix('.policy.jsonl')

    def test_binds_real_requests_and_rejects_training_approval(self):
        with tempfile.TemporaryDirectory() as td:
            paths=self.make_trace(td); result=validate(*paths)
            self.assertEqual(result['actual_infer_attempts'],2)
            self.assertEqual(result['causal_valid_requests'],1)
            self.assertEqual(result['last_request_step'],25)
            self.assertEqual(result['status'],'diagnostic_only_not_training_approved')
            self.assertIn('rgb_hdf5',result['missing_training_artifacts'])

    def test_out_of_sync_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            raw,obs,policy=self.make_trace(td)
            rows=[json.loads(x) for x in policy.read_text().splitlines()]
            rows[2]['physics_step']=24
            policy.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            with self.assertRaisesRegex(ValueError,'reordered policy request'):
                validate(raw,obs,policy)

    def test_bad_clock_and_pre_camera_request_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            raw,obs,policy=self.make_trace(td)
            original=policy.read_text();rows=[json.loads(x) for x in original.splitlines()]
            rows[1]['request_monotonic_ns']=0
            policy.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            with self.assertRaisesRegex(ValueError,'precedes its camera'):
                validate(raw,obs,policy)
            policy.write_text(original)
            r=[json.loads(x) for x in raw.read_text().splitlines()]
            r[0]['clock_origin']='first_instrumented_expert_step_after_setup_stability'
            raw.write_text('\n'.join(json.dumps(x) for x in r)+'\n')
            with self.assertRaisesRegex(ValueError,'clock origin'):
                validate(raw,obs,policy)


if __name__=='__main__':unittest.main()

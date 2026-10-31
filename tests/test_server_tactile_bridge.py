import unittest
import numpy as np
import torch
from server_tactile_bridge import capability_metadata,prepare_input,stack_tactile_prepared
from tactile_rpc import encode_tactile_window
from tests.test_tactile_rpc import window


class StrictVisualTransform:
    """Simulates old LingBot transform: reject any unknown tactile keys."""
    def apply(self, observation,policy_eval=False):
        assert policy_eval
        assert not any(k.startswith('observation.tactile.') for k in observation)
        return {'images':torch.zeros(3,3,32,32),
                'state':torch.as_tensor(observation['observation.state']).float()}


class ServerTactileBridgeTests(unittest.TestCase):
    def test_server_capability_not_advertised_on_unwired_model(self):
        self.assertEqual(capability_metadata(enabled=False),{})
        for kwargs in ({'enabled':True},
                       {'enabled':True,'model_wired':True},
                       {'enabled':True,'adapter_checkpoint_sha256':'0'*64}):
            with self.assertRaisesRegex(RuntimeError,'cannot advertise'):
                capability_metadata(**kwargs)
        meta=capability_metadata(enabled=True,model_wired=True,
                                 adapter_checkpoint_sha256='a'*64)
        self.assertIn('tactile_v1',meta['capabilities'])

    def test_old_transform_cannot_silently_drop_tactile_fields(self):
        source={'observation.state':np.zeros(14,dtype=np.float32)}
        source.update(encode_tactile_window(window()))
        prepared=prepare_input(source,StrictVisualTransform(),tactile_enabled=True)
        self.assertEqual(prepared['tactile_history'].shape,(5,4,3))
        self.assertEqual(prepared['tactile_mask'].shape,(5,4))
        self.assertTrue(prepared['tactile_valid'][0])
        self.assertEqual(prepared['state'].shape,(14,))
        with self.assertRaisesRegex(RuntimeError,'disabled'):
            prepare_input(source,StrictVisualTransform(),tactile_enabled=False)

    def test_mixed_client_batch_uses_explicit_invalid_sensor_fallback(self):
        transform=StrictVisualTransform()
        good={'observation.state':np.zeros(14,dtype=np.float32),**encode_tactile_window(window())}
        old={'observation.state':np.ones(14,dtype=np.float32)}
        prepared=[prepare_input(msg,transform,tactile_enabled=True) for msg in (good,old)]
        batch=stack_tactile_prepared(prepared)
        self.assertEqual(tuple(batch['tactile_history'].shape),(2,5,4,3))
        self.assertTrue(batch['tactile_valid'][0,0])
        self.assertFalse(batch['tactile_valid'][1,0])
        with self.assertRaisesRegex(ValueError,'mixed'):
            stack_tactile_prepared([prepared[0], {'state':torch.zeros(14)}])


if __name__=='__main__':unittest.main()

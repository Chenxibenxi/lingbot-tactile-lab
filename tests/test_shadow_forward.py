"""No GPU/model weights: prove physics→transport→state-token CPU contract."""
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from force_probe import ForceProbe
from shadow_forward import build_shadow_request, shadow_state_token
from tactile_adapter import TactileStateResidual
from tests.test_force_probe import fake_robot, contact


class ShadowForwardTests(unittest.TestCase):
    def test_same_probe_causal_payload_reaches_gated_state_token(self):
        robot=fake_robot()
        scene=NS(get_contacts=lambda:[contact('fl_link7','cube',[.004,0,0])])
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'TACTILE_LIVE_WINDOW':'1'}):
            probe=ForceProbe(td,task='lift_pot',seed=2,episode=0,dt=.004,robot=robot)
            vision={'observation.state':np.zeros(14,dtype=np.float32),'task':'lift the pot'}
            self.assertIs(build_shadow_request(vision,probe,enabled=False),vision)
            for _ in range(25): probe.sample(scene,robot,1)
            request=build_shadow_request(vision,probe,enabled=True)
            self.assertEqual(request['observation.tactile.history'].shape,(5,4,3))
            adapter=TactileStateResidual(state_width=12,hidden_width=8)
            token=torch.ones(1,12)
            self.assertTrue(torch.equal(shadow_state_token(token,request,adapter),token))
            with torch.no_grad(): adapter.to_state.weight.fill_(.1)
            self.assertFalse(torch.equal(shadow_state_token(token,request,adapter),token))
            self.assertTrue(torch.equal(shadow_state_token(token,vision,adapter),token))
            probe.close()

    def test_initial_incomplete_force_history_is_explicit_visual_fallback(self):
        robot=fake_robot()
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'TACTILE_LIVE_WINDOW':'1'}):
            probe=ForceProbe(td,task='lift_pot',seed=2,episode=0,dt=.004,robot=robot)
            request=build_shadow_request({'task':'test'},probe,enabled=True)
            self.assertFalse(request['observation.tactile.valid'][0])
            adapter=TactileStateResidual(12)
            with torch.no_grad(): adapter.to_state.weight.fill_(.1)
            baseline=torch.ones(1,12)
            self.assertTrue(torch.equal(shadow_state_token(baseline,request,adapter),baseline))


if __name__=='__main__': unittest.main()

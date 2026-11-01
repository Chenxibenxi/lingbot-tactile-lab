import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from adapter_artifact_gate import sha256
from tactile_adapter import TactileStateResidual
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
        with self.assertRaisesRegex(RuntimeError,'verifiable approved relay training artifact'):
            capability_metadata(enabled=True,model_wired=True,
                                adapter_checkpoint_sha256='a'*64)  # fake SHA cannot enable real service

    def test_only_synthetic_fully_approved_artifacts_can_advertise_not_real_candidate(self):
        # Temporary synthetic fixture proves the positive *code path*, not an
        # actual trained model or a real owner approval.
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)
            for folder in ('models','datasets','configs'):(repo/folder).mkdir()
            model=TactileStateResidual(8)
            with torch.no_grad():model.to_state.weight.fill_(.01)
            weight=repo/'models/trained.safetensors';save_file(model.state_dict(),str(weight))
            registry={'schema':'lingbot_tactile_dataset_registry_v1',
                      'approved_paired_episodes':['train_seed']}
            split={'schema':'lingbot_tactile_seed_split_plan_v1',
                   'source_machine_role':'relay_single_rtx3090','status':'approved_for_training',
                   'train':['train_seed'],'validation':['val_seed'],'test':['test_seed']}
            registry_path=repo/'datasets/provenance_v1.json';registry_path.write_text(json.dumps(registry))
            split_path=repo/'datasets/split.json';split_path.write_text(json.dumps(split))
            norm={'schema':'lingbot_tactile_train_norm_v1','status':'fit_on_train_split_only',
                  'source_trials':['train_seed'],'training_approval':True,
                  'median':[[0.,0.,0.] for _ in range(4)],
                  'scale_iqr_floor_0p1':[[1.,1.,1.] for _ in range(4)],
                  'split_candidate_sha256':sha256(split_path)}
            norm_path=repo/'configs/norm.json';norm_path.write_text(json.dumps(norm))
            manifest={'schema':'lingbot_tactile_deployment_evidence_v1',
                      'status':'approved_for_deployment','source_machine_role':'relay_single_rtx3090',
                      'tactile_conditioned':True,'trained_optimizer_steps':1,
                      'base_checkpoint_sha256':'b'*64,'adapter_sha256':sha256(weight),
                      'provenance_sha256':sha256(registry_path),
                      'split_sha256':sha256(split_path),'normalization_sha256':sha256(norm_path),
                      'split_path':'datasets/split.json','normalization_path':'configs/norm.json'}
            evidence=repo/'models/approved.json';evidence.write_text(json.dumps(manifest))
            args={'enabled':True,'model_wired':True,'repo_root':repo,
                  'adapter_checkpoint_path':'models/trained.safetensors',
                  'deployment_manifest_path':'models/approved.json',
                  'adapter_checkpoint_sha256':sha256(weight)}
            meta=capability_metadata(**args)
            self.assertIn('tactile_v1',meta['capabilities'])
            self.assertEqual(meta['tactile_base_checkpoint_sha256'],'b'*64)
            norm['median'][0][0]=.3;norm_path.write_text(json.dumps(norm))
            manifest['normalization_sha256']=sha256(norm_path);evidence.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError,'verifiable approved'):
                capability_metadata(**args)  # signed config differs from persisted buffers
            norm['median'][0][0]=0.;norm_path.write_text(json.dumps(norm))
            manifest['normalization_sha256']=sha256(norm_path);evidence.write_text(json.dumps(manifest))
            norm['training_approval']=False;norm_path.write_text(json.dumps(norm))
            with self.assertRaisesRegex(RuntimeError,'verifiable approved'):
                capability_metadata(**args)
            norm['training_approval']=True;norm_path.write_text(json.dumps(norm))
            registry['approved_paired_episodes']=[];registry_path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(RuntimeError,'verifiable approved'):
                capability_metadata(**args)
            registry['approved_paired_episodes']=['train_seed'];registry_path.write_text(json.dumps(registry))
            with torch.no_grad():model.to_state.weight.zero_()
            zero=repo/'models/zero_output.safetensors';save_file(model.state_dict(),str(zero))
            manifest['adapter_sha256']=sha256(zero);evidence.write_text(json.dumps(manifest))
            zero_args=dict(args,adapter_checkpoint_path='models/zero_output.safetensors',
                           adapter_checkpoint_sha256=sha256(zero))
            with self.assertRaisesRegex(RuntimeError,'verifiable approved'):
                capability_metadata(**zero_args)

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

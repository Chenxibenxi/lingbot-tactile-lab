import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import record_recovery_development as collector
from validate_expert_data import sha256


class CollectorPreflight(unittest.TestCase):
    def test_source_must_match_attempt_and_full_action_span(self):
        source=dict(task='pick_diverse_bottles',seed=100119,episode=0,physics_steps=100,saved_frames=10)
        execution=dict(physics_steps=100,action_count=10)
        collector.require_attempt_identity(source,execution,100119)
        for key,value in [('seed',100120),('episode',1),('physics_steps',99),('saved_frames',9)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                collector.require_attempt_identity(dict(source,**{key:value}),execution,100119)

    def test_test_seeds_cannot_enter_development(self):
        for seed in (100103,100118,True):
            with self.assertRaises(ValueError):collector.preflight(seed,'expert_control')

    def test_unready_manifest_cannot_reach_gpu_or_simulator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);protocol=root/'protocol.md';protocol.write_text('synthetic fixture')
            (root/'datasets').mkdir();(root/'data/reports/recovery_development_v1').mkdir(parents=True)
            (root/'datasets/recovery_development_reservations_v1.json').write_text(json.dumps({'protocol_sha256':sha256(protocol)}))
            path=root/'data/reports/recovery_development_v1/manifest.json'
            path.write_text(json.dumps({'gpu_execution_ready':False,'protocol_sha256':sha256(protocol)}))
            with patch.object(collector,'R',root),patch.object(collector,'PROTOCOL',protocol):
                with self.assertRaisesRegex(ValueError,'not ready/frozen'):collector.preflight(100119,'expert_control')
                path.write_text(json.dumps({'gpu_execution_ready':True,'protocol_sha256':sha256(protocol),'files':{}}))
                with self.assertRaisesRegex(ValueError,'missing implementation source pins'):collector.preflight(100119,'expert_control')
            self.assertFalse((root/'data/paired').exists())

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import record_recovery_development_v2 as collector
from validate_expert_data import sha256


class CollectorTest(unittest.TestCase):
    def test_exclusive_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            collector.write_new(path, {'training_approval':False})
            with self.assertRaises(FileExistsError): collector.write_new(path, {'training_approval':True})
            self.assertFalse(json.loads(path.read_text())['training_approval'])

    def test_unready_manifest_rejected_before_gpu_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'datasets').mkdir()
            protocol = root/'protocol.md'; protocol.write_text('test protocol')
            collector.write_new(root/'datasets/recovery_development_reservations_v2.json',
                dict(seeds=[dict(seed=100123)],protocol_sha256=sha256(protocol)))
            report = root/'data/reports/recovery_development_v2'; report.mkdir(parents=True)
            collector.write_new(report/'manifest.json', dict(gpu_execution_ready=False,protocol_sha256=sha256(protocol)))
            with patch.object(collector,'R',root), patch.object(collector,'PROTOCOL',protocol):
                with self.assertRaisesRegex(ValueError,'not frozen'):
                    collector.preflight(100123,'prefix_generation')
                with self.assertRaisesRegex(ValueError,'registered'):
                    collector.preflight(100119,'prefix_generation')

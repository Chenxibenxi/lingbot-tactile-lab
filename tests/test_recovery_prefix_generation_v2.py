import copy
from pathlib import Path
import tempfile
import unittest
import test_recovery_pair_state_v2 as state_fixtures
from test_recovery_shared_prefix_v2 import fixture
from recovery_prefix_generation_v2 import generate_prefix, seal_prefix
from recovery_shared_prefix_v2 import load_prefix


class GenerationTest(unittest.TestCase):
    def generate(self, fail=False):
        state = state_fixtures.StateTest(); state.setUp()
        clock = [0]; stages = []; events = []
        def snapshot():
            result = copy.deepcopy(state.state); result['physics_step'] = clock[0]; return result
        def commands(stage):
            stages.append(stage)
            yield [0.]*14
        def execute(command):
            if fail: raise RuntimeError('controller failed')
            clock[0] += 10
        doc = generate_prefix(identity=fixture()['identity'], commands=commands,
            read_clock=lambda:clock[0], read_state=snapshot,
            read_rgb_hashes=lambda:{k:'d'*64 for k in ('head_camera','left_camera','right_camera')},
            execute=execute, emit=events.append)
        self.assertEqual(stages, ['grasp','lift'])
        self.assertEqual(len(events), 4)
        return doc

    def test_generate_seal_and_refuse_overwrite(self):
        doc = self.generate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prefix.json'
            receipt = seal_prefix(path, doc, expected_identity=doc['identity'])
            self.assertEqual(load_prefix(path, expected_sha256=receipt['sha256'], expected_identity=doc['identity']), doc)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): seal_prefix(path, doc, expected_identity=doc['identity'])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_failed_execution_cannot_return_prefix(self):
        with self.assertRaises(RuntimeError): self.generate(fail=True)

    def test_incomplete_reference_cannot_be_sealed(self):
        doc = self.generate()
        del doc['boundary_rgb_sha256']['head_camera']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prefix.json'
            with self.assertRaises(ValueError): seal_prefix(path,doc,expected_identity=doc['identity'])
            self.assertFalse(path.exists())

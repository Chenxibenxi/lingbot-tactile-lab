import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from recovery_shared_prefix_v2 import replay_verified_prefix
from test_recovery_shared_prefix_v2 import fixture
import test_recovery_pair_state_v2 as state_fixtures


class VerifiedPrefixTest(unittest.TestCase):
    def run_case(self, failure=None):
        doc = fixture()
        states = state_fixtures.StateTest(); states.setUp()
        for name, clock in (('initial',0), ('boundary',30)):
            doc[name+'_state'] = copy.deepcopy(states.state)
            doc[name+'_state']['physics_step'] = clock
            doc[name+'_rgb_sha256'] = {key:'d'*64 for key in ('head_camera','left_camera','right_camera')}
        if failure == 'bad_reference':
            doc['boundary_state']['physics_step'] = 29
        clock = [0]; events = []; commands = []
        def read_state():
            state = copy.deepcopy(doc['initial_state' if clock[0] == 0 else 'boundary_state'])
            if failure == ('initial' if clock[0] == 0 else 'boundary'):
                state['values']['bottle1.position'][0] += .01
            return state
        def read_rgb():
            rgb = dict(doc['initial_rgb_sha256'])
            if failure == 'rgb': rgb['head_camera'] = 'e'*64
            return rgb
        def execute(command):
            commands.append(command); clock[0] += 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prefix.json'
            raw = json.dumps(doc).encode(); path.write_bytes(raw)
            args = dict(expected_sha256=hashlib.sha256(raw).hexdigest(), expected_identity=doc['identity'],
                        read_clock=lambda:clock[0], read_state=read_state, read_rgb_hashes=read_rgb,
                        execute=execute, emit=events.append)
            if failure:
                with self.assertRaises(ValueError): replay_verified_prefix(path, **args)
            else:
                result = replay_verified_prefix(path, **args)
                self.assertEqual(result['status'], 'shared_prefix_verified')
                self.assertFalse(result['training_approval'])
        return commands, events

    def test_success(self):
        commands, events = self.run_case()
        self.assertEqual(len(commands), 3)
        self.assertEqual(events[-1]['event'], 'boundary_state_audit')

    def test_initial_mismatch_executes_nothing(self):
        for failure in ('initial','rgb','bad_reference'):
            with self.subTest(failure=failure):
                commands, _ = self.run_case(failure)
                self.assertEqual(commands, [])

    def test_boundary_mismatch_preserves_failed_audit(self):
        commands, events = self.run_case('boundary')
        self.assertEqual(len(commands), 3)
        self.assertFalse(events[-1]['observable_state_equal'])

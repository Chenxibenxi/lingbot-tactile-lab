import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from recovery_shared_prefix_v2 import load_prefix, replay_prefix, validate_prefix


def fixture():
    identity = dict(seed=123, asset_sha256='a'*64, controller_sha256='b'*64, protocol_sha256='c'*64)
    rows = [dict(action_index=i, source_role='expert', phase=phase,
                 before_physics_step=i*10, after_physics_step=(i+1)*10,
                 command_state_14=[0.0]*14, training_approval=False)
            for i, phase in enumerate(('grasp', 'grasp', 'lift'))]
    return dict(schema='recovery_shared_prefix_v2', identity=identity, rows=rows,
                boundary='before_first_place', training_approval=False)


class PrefixTest(unittest.TestCase):
    def test_seal_and_identity(self):
        doc = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prefix.json'
            raw = json.dumps(doc).encode()
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(load_prefix(path, expected_sha256=digest, expected_identity=doc['identity']), doc)
            changed = dict(doc['identity'], seed=124)
            with self.assertRaises(ValueError):
                load_prefix(path, expected_sha256=digest, expected_identity=changed)
            path.write_bytes(raw+b' ')
            with self.assertRaises(ValueError):
                load_prefix(path, expected_sha256=digest, expected_identity=doc['identity'])

    def test_invalid_prefixes(self):
        cases = []
        for field, value in [('action_index', 7), ('phase', 'place'),
                             ('source_role', 'external_intervention'), ('before_physics_step', 9),
                             ('after_physics_step', True), ('training_approval', True),
                             ('command_state_14', [float('nan')]*14)]:
            doc = fixture(); doc['rows'][1][field] = value; cases.append(doc)
        doc = fixture(); del doc['rows'][1]; cases.append(doc)
        doc = fixture(); doc['rows'][0]['phase'] = 'lift'; cases.append(doc)
        doc = fixture(); doc['rows'][2]['phase'] = 'grasp'; cases.append(doc)
        for doc in cases:
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                validate_prefix(doc, expected_identity=doc['identity'])

    def test_replay_and_no_training_promotion(self):
        doc = fixture(); clock = [0]; events = []; commands = []
        def execute(command):
            commands.append(list(command)); clock[0] += 10
            doc['rows'][-1]['command_state_14'][0] = 99
        result = replay_prefix(doc, expected_identity=copy.deepcopy(doc['identity']),
                               read_clock=lambda: clock[0], execute=execute, emit=events.append)
        self.assertEqual(commands, [[0.0]*14]*3)
        self.assertEqual(len(events), 6)
        self.assertFalse(result['training_approval'])
        self.assertTrue(result['boundary_state_audit_required'])

    def test_drift_stops_before_next_action(self):
        doc = fixture(); clock = [0]; events = []
        def execute(command): clock[0] += 11
        with self.assertRaises(ValueError):
            replay_prefix(doc, expected_identity=doc['identity'], read_clock=lambda: clock[0],
                          execute=execute, emit=events.append)
        self.assertEqual([e['event'] for e in events], ['before_prefix_action'])

    def test_controller_exception_keeps_attempt_only(self):
        doc = fixture(); events = []
        def execute(command): raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            replay_prefix(doc, expected_identity=doc['identity'], read_clock=lambda: 0,
                          execute=execute, emit=events.append)
        self.assertEqual([e['event'] for e in events], ['before_prefix_action'])

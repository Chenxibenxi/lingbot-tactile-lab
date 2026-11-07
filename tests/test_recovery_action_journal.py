import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from recovery_action_journal import RecoveryJournal
from recovery_expert_executor import execute_recovery
from test_recovery_expert_executor import Task,capture,proposals


class JournalTest(unittest.TestCase):
    def test_controller_exception_preserves_attempt_not_fake_execution(self):
        class Broken(Task):
            def take_action(self,*args,**kwargs):raise RuntimeError('controller interrupted')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'journal.jsonl'
            with RecoveryJournal(path,seed=100119,condition='left_open_once_recovery',protocol_sha256='a'*64) as journal:
                with patch('recovery_expert_executor.online_expert_proposals',side_effect=proposals):
                    with self.assertRaises(RuntimeError):execute_recovery(Broken(),capture,perturb=True,on_event=journal.emit)
            events=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([x['event'] for x in events],['journal_header','before_action'])
            self.assertFalse(any(x.get('official_success') for x in events))
            with self.assertRaises(FileExistsError):RecoveryJournal(path,seed=100119,condition='expert_control',protocol_sha256='a'*64)

    def test_successful_execution_has_paired_markers(self):
        events=[]
        with patch('recovery_expert_executor.online_expert_proposals',side_effect=proposals):
            result=execute_recovery(Task(),capture,perturb=True,on_event=events.append)
        self.assertEqual([x['event'] for x in events],['before_action','after_action']*6+['execution_finished'])
        self.assertEqual(result['action_count'],6)

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_recovery_development_batch_v2 as batch


class BatchTest(unittest.TestCase):
    def test_prefix_failure_skips_branches_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); report=root/'data/reports/recovery_development_v2';report.mkdir(parents=True)
            (root/'data/paired').mkdir(parents=True)
            def launch(args,**kwargs):
                seed,kind=args[-2:]
                trial=root/f'data/paired/recovery_v2_{seed}_{kind}';trial.mkdir()
                (trial/'failure.json').write_text(json.dumps(dict(status='needs_inspection',kind=kind,identity=dict(seed=int(seed)))))
                return SimpleNamespace(returncode=1)
            with patch.object(batch,'R',root), patch.object(batch,'verify_manifest',return_value={'seeds':[100123]}), \
                 patch.object(batch,'resource_status',return_value='available'), \
                 patch.object(batch.subprocess,'run',side_effect=launch) as run, patch.object(batch,'report'):
                batch.run()
                self.assertEqual(run.call_count,1)
                data=json.loads((report/'queue_result.json').read_text())
                self.assertEqual([x['status'] for x in data['attempts']],['needs_inspection','prerequisite_failed','prerequisite_failed'])
                with self.assertRaises(FileExistsError):batch.run()
                self.assertEqual(run.call_count,1)

    def test_busy_gpu_launches_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); report=root/'data/reports/recovery_development_v2';report.mkdir(parents=True)
            with patch.object(batch,'R',root), patch.object(batch,'verify_manifest',return_value={'seeds':[100123]}), \
                 patch.object(batch,'resource_status',return_value='gpu_busy'), patch.object(batch.subprocess,'run') as run:
                batch.run();run.assert_not_called()
            self.assertEqual(json.loads((report/'queue_result.json').read_text())['reason'],'gpu_busy')

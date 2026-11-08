import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import prepare_feedback_confirmation_v3 as freeze

class Freeze(unittest.TestCase):
    def test_missing_or_failed_pilot_creates_no_formal_admission(self):
        for failed in (False,True):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/'docs').mkdir();(root/'data/reports').mkdir(parents=True)
                if failed:
                    prior=root/'data/reports/controlled_feedback_pilot_v3_final';prior.mkdir()
                    (prior/'summary.json').write_text(json.dumps({'technical_gate':False}))
                with patch.object(freeze,'R',root),patch.object(freeze,'audit',return_value={'technical_gate':False}),patch.object(freeze,'measure') as capacity:
                    with self.assertRaises((FileNotFoundError,ValueError)):freeze.prepare()
                    capacity.assert_not_called()
                self.assertFalse((root/'data/reports/feedback_confirmation_v3').exists())
                self.assertFalse((root/'docs/CONTROLLED_FEEDBACK_CONFIRMATION_PROTOCOL_V2_ZH.md').exists())
    def test_existing_freeze_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data/reports/feedback_confirmation_v3').mkdir(parents=True)
            with patch.object(freeze,'R',root):
                with self.assertRaises(FileExistsError):freeze.prepare()

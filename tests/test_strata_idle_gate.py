"""Honor authoritative busy/queued state despite stale last-request phase."""
import io
import json
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout,redirect_stderr
from strata_idle_gate import require_idle,main


class StrataIdleGateTests(unittest.TestCase):
    def test_idle_legacy_and_explicit_idle_pass(self):
        for status in ({'busy':False,'queued':0},
                       {'busy':False,'queued':0,'phase':'idle'},
                       {'busy':False,'queued':0,'phase':'unloaded'}):
            self.assertTrue(require_idle(status))

    def test_stale_thinking_phase_is_ignored_only_after_busy_false(self):
        # Real Strata leaves the last phase in /status when request busy ends.
        for phase in ('thinking','answering','generating'):
            self.assertTrue(require_idle({'busy':False,'queued':0,'phase':phase}))

    def test_busy_or_queued_refused_even_when_phase_looks_idle(self):
        for status in ({'busy':True,'queued':0,'phase':'idle'},
                       {'busy':False,'queued':1,'phase':'idle'},
                       {'busy':False,'queued':'0','phase':'idle'},
                       {'queued':0,'phase':'idle'}):
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError,'Strata'):
                    require_idle(status)

    def test_cli_hides_private_status_field_and_blocks_real_busy_request(self):
        stdout=io.StringIO();stderr=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps({'busy':True,'queued':0,'phase':'thinking','prompt':'private'}))), \
             redirect_stdout(stdout),redirect_stderr(stderr):
            self.assertEqual(main(),4)
        self.assertNotIn('private',stdout.getvalue()+stderr.getvalue())
        self.assertIn('REFUSED',stderr.getvalue())


if __name__=='__main__':unittest.main()

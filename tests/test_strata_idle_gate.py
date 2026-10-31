"""Do not evict Strata while it reports an active generation phase."""
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

    def test_stale_false_busy_does_not_override_thinking_or_queued(self):
        for status in ({'busy':False,'queued':0,'phase':'thinking'},
                       {'busy':False,'queued':0,'phase':'generating'},
                       {'busy':False,'queued':1},
                       {'busy':True,'queued':0},
                       {'busy':False,'queued':'0'}):
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError,'Strata'):
                    require_idle(status)

    def test_cli_refuses_thinking_without_exposing_prompt(self):
        stdout=io.StringIO();stderr=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps({'busy':False,'queued':0,'phase':'thinking','prompt':'private'}))), \
             redirect_stdout(stdout),redirect_stderr(stderr):
            self.assertEqual(main(),4)
        self.assertNotIn('private',stdout.getvalue()+stderr.getvalue())
        self.assertIn('REFUSED',stderr.getvalue())


if __name__=='__main__':unittest.main()

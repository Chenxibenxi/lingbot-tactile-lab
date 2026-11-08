import unittest
from types import SimpleNamespace
from gpu_resource_snapshot import snapshot
from dispatch_resume_contract import remaining
class Resource(unittest.TestCase):
    def test_real_probe_interfaces_are_read_only(self):
        calls=[]
        def run(args,**kw):
            calls.append(args);self.assertEqual(kw['timeout'],10)
            return SimpleNamespace(stdout='{"state":"free","waiting":[]}' if '--json' in args else '')
        x=snapshot(run);self.assertTrue(x['probe_ok']);self.assertEqual(len(calls),2)
    def test_probe_errors_fail_closed(self):
        self.assertFalse(snapshot(lambda *a,**kw:SimpleNamespace(stdout='bad'))['probe_ok'])
    def test_resume_only_completed_prefix(self):
        ok=dict(status='completed',returncode=0,resource_censored=False)
        self.assertEqual(remaining(['a','b'],{'a':ok},{'a'}),['b'])
        for receipts,started in [({}, {'a'}),({'b':ok},{'b'}),({'a':dict(ok,returncode=1)},{'a'}),({'a':ok},set())]:
            with self.assertRaises(ValueError):remaining(['a','b'],receipts,started)

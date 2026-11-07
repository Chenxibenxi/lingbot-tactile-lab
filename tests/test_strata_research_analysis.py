import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import strata_research_analysis as client


class Response:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'仅为模拟测试分析。'}}]}).encode()


class StrataAnalysisTest(unittest.TestCase):
    def test_unloaded_does_not_request_or_acquire(self):
        with tempfile.TemporaryDirectory() as d, patch.object(client,'get',return_value={'service':'strata','loaded':False}), patch.object(client,'urlopen') as http,patch.object(client.subprocess,'run') as lease:
            with self.assertRaises(RuntimeError):client.analyse(Path(d)/'missing',[],Path(d)/'out')
            http.assert_not_called();lease.assert_not_called()

    def test_complete_response_and_own_lease_release(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'report.md').write_text('合成测试证据，不是研究结果。')
            replies=[{'service':'strata','loaded':True},{'data':[{'id':'mock-model'}]},{'loaded':True}]
            def lease(args,**kwargs):
                return SimpleNamespace(stdout=json.dumps({'holder':client.OWNER,'lease':{'pid':os.getpid()}}))
            with patch.object(client,'get',side_effect=replies),patch.object(client,'urlopen',return_value=Response()) as http,patch.object(client.subprocess,'run',side_effect=lease) as calls:
                client.analyse(root/'report.md',[],root/'out')
                self.assertTrue((root/'out/analysis.md').exists())
                self.assertEqual(http.call_args.args[0].full_url,'http://127.0.0.1:8080/v1/chat/completions')
                self.assertEqual([c.args[0][2] for c in calls.call_args_list],['acquire','status','release'])

    def test_does_not_release_another_lease(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'report.md').write_text('合成测试')
            replies=[{'service':'strata','loaded':True},{'data':[{'id':'mock-model'}]},{'loaded':True}]
            with patch.object(client,'get',side_effect=replies),patch.object(client,'urlopen',return_value=Response()),patch.object(client.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps({'holder':'other','lease':{'pid':123}}))) as calls:
                client.analyse(root/'report.md',[],root/'out')
                self.assertEqual([c.args[0][2] for c in calls.call_args_list],['acquire','status'])

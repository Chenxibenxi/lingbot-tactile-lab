import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import report_m7_training_summary as report
class SummaryTest(unittest.TestCase):
 def fixture(self,root):
  r=Path(root);seeds=[str(s) for s in (100049,100067,100068,100069,100070)]
  for i in range(3):
   n=r/f'data/reports/cwmj_m7_training_r{i}';o=r/f'data/reports/cwmj_m6_training_r{i}';n.mkdir(parents=True);o.mkdir(parents=True)
   w=n/'weight';w.write_text('fixture')
   p=dict(initialization=dict(parameter_sha256=str(i)),transformed_dataset_sha256='same')
   for d in (n,o):(d/'protocol.json').write_text(json.dumps(p))
   values={s:{'True':.1,'False':.2} for s in seeds}
   (o/'result.json').write_text(json.dumps(dict(results={'frozen':[1,2]},summary_first10_mae={a:values for a in ('zero','true','mismatch')})))
   (n/'result.json').write_text(json.dumps(dict(status='completed_real_adapter_training_offline_pilot',repeat_index=i,results={'frozen':[1,2]},updates={'contact_state':dict(steps=720,artifact=str(w),sha256=report.sha256(w))},summary_first10_mae={'contact_state':values})))
  return r
 def test_all_repeats_and_no_benefit_claim(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.fixture(d)
   with patch.object(report,'R',r),patch.object(report,'DEST',r/'published'):report.report()
   s=json.loads((r/'published/summary.json').read_text());self.assertEqual(len(s['records']),3);self.assertFalse(s['closed_loop_benefit_proven'])
 def test_missing_repeat_and_changed_weight_block_publication(self):
  for mode in ('missing','changed'):
   with tempfile.TemporaryDirectory() as d:
    r=self.fixture(d)
    if mode=='missing':(r/'data/reports/cwmj_m7_training_r2/result.json').unlink()
    else:(r/'data/reports/cwmj_m7_training_r2/weight').write_text('changed')
    with patch.object(report,'R',r),patch.object(report,'DEST',r/'published'):
     with self.assertRaises((FileNotFoundError,ValueError)):report.report()
    self.assertFalse((r/'published/summary.json').exists())
if __name__=='__main__':unittest.main()

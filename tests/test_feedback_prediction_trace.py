import json,tempfile,unittest
from pathlib import Path
import numpy as np
from feedback_prediction_trace import PredictionTrace
class TraceTest(unittest.TestCase):
 def test_prefix_clock_and_copy(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'trace.jsonl';w=PredictionTrace(p);x=np.zeros((50,14));w.request(index=0,physics_step=0,commands=x,physical_state={})
   x[0]=9
   with self.assertRaises(ValueError):w.action(command=x[0],physics_before=0,physics_after=1,before={},after={})
   with self.assertRaises(ValueError):w.action(command=np.zeros(14),physics_before=1,physics_after=2,before={},after={})
   w.action(command=np.zeros(14),physics_before=0,physics_after=2,before={},after={})
   w.request(index=1,physics_step=2,commands=x,physical_state={})
   w.action(command=x[0],physics_before=2,physics_after=3,before={},after={});w.close(complete=True)
   rows=[json.loads(l) for l in p.read_text().splitlines()];self.assertEqual(rows[1]['commands'][0],[0]*14);self.assertTrue(rows[-1]['complete'])
   with self.assertRaises(FileExistsError):PredictionTrace(p)
 def test_reject_nonfinite(self):
  with tempfile.TemporaryDirectory() as d:
   w=PredictionTrace(Path(d)/'x');x=np.zeros((50,14));x[1,1]=np.nan
   with self.assertRaises(ValueError):w.request(index=0,physics_step=0,commands=x,physical_state={})
   w.close()
if __name__=='__main__':unittest.main()

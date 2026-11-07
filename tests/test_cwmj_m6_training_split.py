import unittest
from prepare_cwmj_m6_training_manifest import validate_split
class TrainingSplit(unittest.TestCase):
 def test_preregistered_merge_and_leakage(self):
  nt=[100058,100062,100064,100065];nv=[100067,100068];t=[100045,100046]+nt;v=[100049]+nv
  validate_split(t,v,nt,nv)
  for bad in [t+[100049],t+[100071],t+[100058],t[1:]]:
   with self.assertRaises(ValueError):validate_split(bad,v,nt,nv)
 def test_insufficient_new_or_test_seed_refused(self):
  for nt,nv in [([100058,100062,100064],[100067,100068]),([100058,100062,100064,100065],[100071,100072])]:
   with self.assertRaises(ValueError):validate_split([100045,100046]+nt,[100049]+nv,nt,nv)

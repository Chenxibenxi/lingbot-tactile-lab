import unittest,json,hashlib,tempfile
from pathlib import Path
from prepare_cwmj_m6 import FOLDS,generate,REPO
from cwmj_m6_samples import verify_collection_manifest
class M6DataContract(unittest.TestCase):
 def test_heldout_and_prior_seeds_refused(self):
  self.assertEqual({s for s,f in FOLDS.items() if f=='train'},set(range(100055,100067)))
  self.assertEqual({s for s,f in FOLDS.items() if f=='validation'},set(range(100067,100071)))
  for seed in [100045,100049,100051,100054,100071,100086]:
   with self.assertRaises(ValueError):generate(seed)
 def test_provenance_mutation_refused(self):
  seed=100067
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);files=generate(seed)
   for n,s in files.items():(p/n).write_text(s)
   manifest={'seed':seed,'fold':'validation','stride':20,'collection_protocol_sha256':hashlib.sha256((REPO/'docs/CWMJ_M6_DATA_EXPANSION_PLAN_ZH.md').read_bytes()).hexdigest(),'files':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()}}
   def put(m):(p/'source_manifest.json').write_text(json.dumps(m))
   put(manifest);verify_collection_manifest(p,seed)
   for key,bad in [('fold','train'),('seed',100055),('stride',10),('collection_protocol_sha256','0'*64)]:
    put(dict(manifest,**{key:bad}))
    with self.assertRaises(ValueError):verify_collection_manifest(p,seed)
   put(manifest);(p/'record_cwmj_m6.py').write_text('changed')
   with self.assertRaises(ValueError):verify_collection_manifest(p,seed)

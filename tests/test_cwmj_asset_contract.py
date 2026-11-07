import unittest,json
from pathlib import Path
import numpy as np
from validate_expert_data import sha256
from cwmj_bottle_factory import ASSET
class AssetContract(unittest.TestCase):
    def test_all_converted_and_source_files_pinned(self):
        m=json.loads((ASSET/'manifest.json').read_text())
        for name,digest in m['converted_files_sha256'].items():self.assertEqual(sha256(ASSET/name),digest)
        for name,digest in m['source_files_sha256'].items():self.assertEqual(sha256(Path(name)),digest)
    def test_positive_physical_values_and_consistent_coordinate_units(self):
        m=json.loads((ASSET/'manifest.json').read_text());d=json.loads((ASSET/'actor_metadata.json').read_text())
        self.assertEqual(d['scale'],[1.,1.,1.]);self.assertLess(max(d['extents']),.3);self.assertGreater(min(d['extents']),.04)
        self.assertAlmostEqual(np.linalg.norm(m['inertia_frame_quaternion_wxyz']),1,places=6)
        inertia=np.array(m['principal_inertia_kg_m2']);self.assertTrue((inertia>0).all());self.assertTrue((2*inertia<=inertia.sum()).all())
        self.assertGreater(m['mass_kg'],0);self.assertFalse(m['training_admitted'])
        self.assertTrue(np.all(np.asarray(d['functional_matrix'])[...,:3,3]<.3))
        self.assertTrue(np.all(np.abs(np.asarray(d['orientation_point'])[:3,3])<.3))

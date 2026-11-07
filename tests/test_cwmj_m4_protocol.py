import unittest
from prepare_cwmj_m4 import generate,FOLDS
class ScenarioProtocol(unittest.TestCase):
    def test_distinct_fixed_scenario_and_source_capture(self):
        self.assertEqual(set(FOLDS),set(range(100045,100051)))
        for seed,fold in FOLDS.items():
            files=generate(seed);pre=files['preregister_cwmj_m4.py'];rec=files['record_cwmj_m4.py']
            self.assertIn(f'relay_pickcwmj_s{seed}_online_h50_stride20_{fold}_expert0',pre)
            self.assertIn("(out/'scenario.json').write_text",rec)
            self.assertIn('asset_evidence=install()',rec)
            self.assertIn("asset_evidence['factory_sha256']",rec)
            self.assertLess(rec.index('asset_evidence=install()'),rec.index('robot.setup_demo('))
    def test_old_scenario_or_future_seed_refused(self):
        for seed in [100023,100036,100041,100051]:
            with self.assertRaises(ValueError):generate(seed)

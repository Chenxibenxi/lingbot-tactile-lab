import json
from pathlib import Path
import tempfile
import unittest

from align_temporal import align
from live_tactile_window import LiveTactileWindow
from tactile_rpc import encode_tactile_window, decode_tactile
from tests.test_align_temporal import fixture


class LiveWindowTests(unittest.TestCase):
    def test_live_matches_offline_at_same_camera_physics_ticks(self):
        with tempfile.TemporaryDirectory() as td:
            raw, obs = fixture(Path(td))
            rows=[json.loads(x) for x in raw.read_text().splitlines()]
            reference=list(align(raw,obs))
            live=LiveTactileWindow(rows[0]['side_links'],rows[0]['dt_s'])
            first=live.snapshot(task='lift_pot',seed=7,episode=0)
            self.assertEqual(first['features'],reference[0]['features'])
            self.assertFalse(first['quality']['window_valid'])
            snapshots=[]
            for row in rows[1:]:
                live.push(row)
                if row['physics_step'] in (25,30):
                    snapshots.append(live.snapshot(task='lift_pot',seed=7,episode=0))
            for candidate, expected in zip(snapshots,reference[1:]):
                for field in ('shape','finger_order','feature_order','features','sample_fraction_mask','quality'):
                    self.assertEqual(candidate[field],expected[field])
                self.assertTrue(decode_tactile(encode_tactile_window(candidate))['valid'][0])

    def test_gapped_stream_is_masked_and_past_anchor_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            raw, obs=fixture(Path(td),missing=(20,))
            rows=[json.loads(x) for x in raw.read_text().splitlines()]
            live=LiveTactileWindow(rows[0]['side_links'],rows[0]['dt_s'])
            for row in rows[1:]:
                if row['physics_step'] > 25: break
                live.push(row)
            value=live.snapshot(task='lift_pot',seed=7,episode=0)
            self.assertFalse(value['quality']['window_valid'])
            self.assertEqual(value['quality']['coverage'],24/25)
            with self.assertRaisesRegex(ValueError,'latest'):
                live.snapshot(task='lift_pot',seed=7,episode=0,anchor_step=20)

    def test_duplicate_live_sample_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            raw,_=fixture(Path(td))
            head,row=[json.loads(x) for x in raw.read_text().splitlines()][:2]
            live=LiveTactileWindow(head['side_links'],head['dt_s'])
            live.push(row)
            with self.assertRaisesRegex(ValueError,'monotonically'):
                live.push(row)


if __name__=='__main__': unittest.main()

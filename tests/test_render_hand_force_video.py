"""Shared noncalibrated force color scale and fail-closed local MP4 scope."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import h5py
from render_hand_force_video import color_proxy,COLOR_TICKS,BGR,render,load_role


class HandForceVideoTests(unittest.TestCase):
    def test_shared_log_color_scale_has_fixed_structural_stress_tiers(self):
        for x,c in zip(COLOR_TICKS,BGR):
            self.assertEqual(color_proxy(x),c)
        self.assertEqual(color_proxy(500),BGR[-1])
        self.assertNotEqual(color_proxy(.1),color_proxy(1.))
        for x in (-1.,math.nan,float('inf')):
            with self.assertRaisesRegex(ValueError,'invalid'):
                color_proxy(x)

    def test_video_cannot_escape_ignored_private_local_report_root(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data/reports').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'data/reports'):
                render(root,root/'docs/leaked.mp4')
            with self.assertRaisesRegex(ValueError,'supported fps'):
                render(root,root/'data/reports/out.mp4',fps=100)

    def test_color_at_capture_does_not_use_future_contact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'data/negative/trial'
            force=path/'force';force.mkdir(parents=True)
            (path/'demonstration/data').mkdir(parents=True)
            source=force/'raw.jsonl'
            header={'dt_s':.004,'side_links':{'left':['fl_link7','fl_link8'],
                                               'right':['fr_link7','fr_link8']}}
            with source.open('w') as fp:
                fp.write(json.dumps(header)+'\n')
                for step in range(1,31):
                    fingers={name:{'load_proxy_n':99. if step==21 and name=='fl_link7'
                                   else .4 if step==11 and name=='fl_link7' else 0.}
                             for name in ('fl_link7','fl_link8')}
                    fp.write(json.dumps({'physics_step':step,'left':{'fingers':fingers},
                        'right':{'fingers':{'fr_link7':{'load_proxy_n':0.},
                                            'fr_link8':{'load_proxy_n':0.}}}})+'\n')
            with source.with_suffix('.frames.jsonl').open('w') as fp:
                fp.write('{}\n')
                for step in (0,10,20,30):fp.write(json.dumps({'physics_step':step})+'\n')
            with h5py.File(path/'demonstration/data/episode0.hdf5','w'):pass
            role=load_role(root,'negative','trial')
            try:
                self.assertEqual(role['peak'][1][0],0.)
                self.assertEqual(role['peak'][2][0],.4)
                self.assertEqual(role['peak'][3][0],99.)
                self.assertEqual(role['instant'][2][0],0.)
            finally:role['hdf5'].close()


if __name__=='__main__':unittest.main()

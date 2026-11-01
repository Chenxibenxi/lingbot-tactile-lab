"""Optional spatial point sidecar must never contaminate base force or policy data."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from force_probe import ForceProbe,gripper_links,sample_contacts
from tests.test_force_probe import fake_robot,body


def touch(a,b,impulse,position):
    return NS(bodies=[body(a),body(b)],points=[NS(impulse=impulse,position=position)])


class SpatialSidecarTests(unittest.TestCase):
    def test_opt_in_separate_private_points_and_unchanged_observation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'trial';force=root/'force';geometry=root/'geometry'
            robot=fake_robot();contacts=[touch('fl_link7','bottle',[.004,0,0],[.1,.2,.3]),
                                          touch('bottle','fr_link8',[.002,0,0],[-.1,.4,.3]),
                                          touch('fr_link7','bottle',[0,0,0],[0,0,0]),
                                          touch('fl_link8','fl_link7',[5,0,0],[0,0,0])]
            class Scene:
                calls=0
                def get_contacts(self):
                    self.calls+=1
                    return contacts if self.calls==1 else []
            scene=Scene()
            with patch.dict(os.environ,{'TACTILE_CONTACT_GEOMETRY_DIR':str(geometry)}):
                probe=ForceProbe(force,task='pick_diverse_bottles',seed=100000,episode=0,dt=.004,robot=robot)
            probe.sample(scene,robot,1);probe.sample(scene,robot,1)
            self.assertFalse(probe.failed);self.assertEqual(scene.calls,2)
            self.assertEqual(set(probe.observation()),{'valid','physics_step','left_load_proxy_n',
                                                        'right_load_proxy_n','left_contact','right_contact'})
            raw=probe.raw_path;probe.close()
            original=[json.loads(x) for x in raw.read_text().splitlines()]
            self.assertEqual([r['physics_step'] for r in original[1:]],[1,2])
            self.assertEqual(original[1]['left']['points'],1)
            self.assertNotIn('position_world_m',raw.read_text())
            private=[json.loads(x) for x in next(geometry.glob('*.geometry.jsonl')).read_text().splitlines()]
            self.assertEqual(private[0]['probe_file'],raw.name)
            self.assertEqual([r['physics_step'] for r in private[1:]],[1,2])
            self.assertEqual(len(private[1]['points']),2)
            self.assertEqual(private[2]['points'],[])
            self.assertEqual(private[1]['points'][0],{'side':'left','finger':'fl_link7',
                'position_world_m':[.1,.2,.3],'load_proxy_n':1.})
            self.assertEqual(private[1]['points'][1]['load_proxy_n'],.5)

    def test_default_absent_and_missing_position_fail_closed(self):
        robot=fake_robot();scene=NS(get_contacts=lambda:[touch('fl_link7','bottle',[.004,0,0],[.1,.2,.3])])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'trial';force=root/'force';geometry=root/'geometry'
            with patch.dict(os.environ,{'TACTILE_CONTACT_GEOMETRY_DIR':str(geometry)}):
                probe=ForceProbe(force,task='pick_diverse_bottles',seed=1,episode=0,dt=.004,robot=robot)
            probe.sample(scene,robot,0);probe.close();self.assertTrue(geometry.exists())
            with patch.dict(os.environ,{},clear=True):
                base=ForceProbe(Path(d)/'base/force',task='pick_diverse_bottles',seed=2,episode=0,dt=.004,robot=robot)
            base.sample(scene,robot,0);base.close();self.assertFalse((Path(d)/'base/geometry').exists())
            bad=NS(get_contacts=lambda:[NS(bodies=[body('fl_link7'),body('bottle')],
                    points=[NS(impulse=[.004,0,0])])])
            with patch.dict(os.environ,{'TACTILE_CONTACT_GEOMETRY_DIR':str(Path(d)/'failed/geometry')}):
                broken=ForceProbe(Path(d)/'failed/force',task='pick_diverse_bottles',seed=3,episode=0,dt=.004,robot=robot)
            broken.sample(bad,robot,0)
            self.assertTrue(broken.failed)
            self.assertIsNone(broken.raw_path)
            self.assertFalse((Path(d)/'failed/geometry').exists())

    def test_geometry_requires_absolute_sibling_and_valid_world_coordinates(self):
        robot=fake_robot()
        with tempfile.TemporaryDirectory() as d:
            force=Path(d)/'trial/force'
            for wrong in ('relative/geometry',str(Path(d)/'other/geometry'),str(Path(d)/'trial/force')):
                with patch.dict(os.environ,{'TACTILE_CONTACT_GEOMETRY_DIR':wrong}):
                    with self.assertRaisesRegex(ValueError,'absolute sibling'):
                        ForceProbe(force,task='pick_diverse_bottles',seed=1,episode=0,dt=.004,robot=robot)
        scene=NS(get_contacts=lambda:[touch('fl_link7','bottle',[.004,0,0],[float('nan'),0,0])])
        with self.assertRaisesRegex(RuntimeError,'finite world position'):
            sample_contacts(scene,gripper_links(robot),.004,geometry_sink=[])
        self.assertEqual(sample_contacts(scene,gripper_links(robot),.004)['left']['points'],1)


if __name__=='__main__':unittest.main()

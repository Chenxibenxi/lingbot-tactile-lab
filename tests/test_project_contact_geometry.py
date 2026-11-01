"""Do not fabricate hand pixel contacts from unrecorded world positions."""
import tempfile
from pathlib import Path
import unittest
import h5py
import numpy as np
from project_contact_geometry import project_world_contact,calibration_audit


K=np.array([[200.,0.,160.],[0.,200.,120.],[0.,0.,1.]])
E=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])


class ContactProjectionPreflightTests(unittest.TestCase):
    def test_project_only_positive_depth_and_inside_frustum(self):
        self.assertEqual(project_world_contact((0,0,1),K,E,(240,320)),(160.,120.,1.))
        self.assertIsNone(project_world_contact((0,0,-1),K,E,(240,320)))
        self.assertIsNone(project_world_contact((1,0,1),K,E,(240,320)))
        self.assertEqual(project_world_contact((-.5,0,1),K,E,(240,320)),(60.,120.,1.))

    def test_per_frame_moving_camera_needs_own_extrinsics(self):
        moved=E.copy();moved[0,3]=.1
        self.assertEqual(project_world_contact((0,0,1),K,moved,(240,320)),(180.,120.,1.))
        self.assertEqual(project_world_contact((0,0,1),K,E,(240,320)),(160.,120.,1.))

    def test_bad_geometry_never_becomes_a_pixel(self):
        for point in ((float('nan'),0,1),(0,0),(0,0,float('inf'))):
            with self.assertRaisesRegex(ValueError,'world contact'):
                project_world_contact(point,K,E,(240,320))
        bad=E.copy();bad[0,0]=2.
        with self.assertRaisesRegex(ValueError,'rotation'):
            project_world_contact((0,0,1),K,bad,(240,320))
        with self.assertRaisesRegex(ValueError,'image shape'):
            project_world_contact((0,0,1),K,E,(0,320))

    def test_calibration_audit_does_not_claim_contact_pixel(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'episode.hdf5'
            with h5py.File(path,'w') as h:
                for cam in ('head_camera','left_camera','right_camera'):
                    group=h.require_group(f'observation/{cam}')
                    group.create_dataset('rgb',data=np.zeros(2,dtype='S1'))
                    group.create_dataset('intrinsic_cv',data=np.stack([K,K]))
                    moved=E.copy();moved[0,3]=.1 if cam!='head_camera' else 0.
                    group.create_dataset('extrinsic_cv',data=np.stack([E,moved]))
            result=calibration_audit(path)
            self.assertFalse(result['may_mark_pixel_contacts'])
            self.assertEqual(result['camera_calibration']['head_camera']['max_extrinsic_translation_component_change_m'],0)
            self.assertEqual(result['camera_calibration']['left_camera']['max_extrinsic_translation_component_change_m'],.1)


if __name__=='__main__':unittest.main()

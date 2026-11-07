"""Project only real same-capture-step points, never paint future geometry."""
import tempfile
from pathlib import Path
import unittest
import h5py
import numpy as np
from render_spatial_contact_video import overlay_same_step_points


class SpatialOverlayTests(unittest.TestCase):
    def test_only_matching_capture_step_projects_hollow_not_claimed_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'calibration.hdf5'
            K=np.array([[200.,0.,160.],[0.,200.,120.],[0.,0.,1.]])
            E=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])
            with h5py.File(p,'w') as f:
                for cam in ('head_camera','left_camera','right_camera'):
                    g=f.require_group('observation/'+cam)
                    g.create_dataset('intrinsic_cv',data=np.stack([K,K]))
                    g.create_dataset('extrinsic_cv',data=np.stack([E,E]))
            with h5py.File(p,'r') as f:
                source={'steps':[0,1],'hdf5':f}
                points=[[{'side':'left','finger':'fl_link7','position_world_m':[0.,0.,1.],
                          'load_proxy_n':.5}]]
                canvas=np.zeros((920,1600,3),dtype=np.uint8)
                self.assertEqual(overlay_same_step_points(canvas,source,0,points),0)
                self.assertEqual(int(canvas.sum()),0)
                self.assertEqual(overlay_same_step_points(canvas,source,1,points),2)
                self.assertGreater(int(canvas.sum()),0)
                edge=np.zeros_like(canvas)
                points[0][0]['position_world_m']=[0.,-.59,1.]
                self.assertEqual(overlay_same_step_points(edge,source,1,points),2)
                self.assertEqual(int(edge[:102].sum()),0)  # no ring spills into header


if __name__=='__main__':unittest.main()

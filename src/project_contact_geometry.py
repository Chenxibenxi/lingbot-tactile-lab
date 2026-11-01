"""CPU-only camera projection preflight for FUTURE private 3D finger-contact logs.

Requires genuine same-physics-step WORLD contact point and per-frame OpenCV
intrinsic/extrinsic calibration. Existing relay captures have calibrated cameras
but omit contact point positions: never invent their pixel coordinates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

CAMERAS=('head_camera','left_camera','right_camera')


def checked_camera(K,E):
    k=np.asarray(K,dtype=np.float64);e=np.asarray(E,dtype=np.float64)
    if k.shape!=(3,3) or e.shape!=(3,4) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(e)):
        raise ValueError('camera calibration shape or finite values invalid')
    if k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],(0.,0.,1.),atol=1e-4):
        raise ValueError('camera intrinsics invalid')
    r=e[:,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-2) or not .99<=np.linalg.det(r)<=1.01:
        raise ValueError('camera extrinsic rotation invalid')
    return k,e


def project_world_contact(point_world,K,E,shape_hw):
    """Return (u,v,depth) inside frustum only; NOT an occlusion/visibility test.

    E is SAPIEN camera.get_extrinsic_matrix() OpenCV world-to-camera (3×4),
    not `cam2world_gl`. Depth sign and frustum must both pass; real RGB
    visibility additionally requires same-capture depth/segmentation evidence.
    """
    k,e=checked_camera(K,E)
    p=np.asarray(point_world,dtype=np.float64)
    if p.shape!=(3,) or not np.all(np.isfinite(p)):
        raise ValueError('world contact position missing or invalid')
    if len(shape_hw)!=2 or any(type(n) not in (int,np.int32,np.int64) or n<=0 for n in shape_hw):
        raise ValueError('image shape invalid')
    height,width=shape_hw
    camera_xyz=e[:,:3]@p+e[:,3]
    z=float(camera_xyz[2])
    if z<=1e-4:return None  # behind camera / on optical center
    pixel=k@camera_xyz
    u,v=float(pixel[0]/pixel[2]),float(pixel[1]/pixel[2])
    if not 0<=u<width or not 0<=v<height:return None
    return (u,v,z)


def calibration_audit(hdf5):
    """Read only per-frame HDF5 calibration; do not claim contact localization."""
    with h5py.File(hdf5,'r') as f:
        report={}
        for cam in CAMERAS:
            prefix=f'observation/{cam}/'
            rgb=f[prefix+'rgb'];intr=f[prefix+'intrinsic_cv'];extr=f[prefix+'extrinsic_cv']
            n=len(rgb)
            if not n or intr.shape!=(n,3,3) or extr.shape!=(n,3,4):
                raise ValueError('same-frame RGB and camera calibration counts/shapes mismatch')
            e0=None;max_translate=0.;max_rotate=0.
            for i in range(n):
                k,e=checked_camera(intr[i],extr[i])
                if e0 is None:e0=e
                max_translate=max(max_translate,float(np.max(np.abs(e[:,3]-e0[:,3]))))
                max_rotate=max(max_rotate,float(np.max(np.abs(e[:,:3]-e0[:,:3]))))
                # Synthetic optical-axis test, not a claim about true contact.
                world=e[:,:3].T @ (np.array([0.,0.,1.])-e[:,3])
                projected=project_world_contact(world,k,e,(240,320))
                if projected is None or not np.allclose(projected[:2],(k[0,2],k[1,2]),atol=1e-3):
                    raise ValueError('camera optical-axis projection invalid')
            report[cam]={'frames_with_calibration':n,
                         'max_extrinsic_translation_component_change_m':round(max_translate,6),
                         'max_extrinsic_rotation_component_change':round(max_rotate,6)}
    return {'schema':'lingbot_contact_geometry_preflight_v1',
            'status':'camera_calibrations_valid_contact_positions_not_recorded',
            'camera_calibration':report,'per_frame_extrinsics_required':True,
            'may_mark_pixel_contacts':False,
            'caution':'Existing ForceProbe records link totals only; future private world contact points must be recorded in the SAME physics run and aligned before projection.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hdf5',required=True,type=Path)
    a=p.parse_args();print(json.dumps(calibration_audit(a.hdf5),indent=2))


if __name__=='__main__':main()

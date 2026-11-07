"""Opt-in in-process asset replacement; no upstream file mutation."""
from pathlib import Path
import json
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]
ASSET=R/'work/cwmj_bottle_asset_v2'

def install():
    import importlib
    import sapien
    import sapien.physx as physx
    from envs.utils.rand_create_actor import rand_pose
    from envs.utils.create_actor import preprocess
    from envs.utils.actor_utils import Actor
    manifest=json.loads((ASSET/'manifest.json').read_text())
    for name,digest in manifest['converted_files_sha256'].items():
        if sha256(ASSET/name)!=digest:raise ValueError('converted asset changed')
    for name,digest in manifest['source_files_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('source asset changed')
    metadata=json.loads((ASSET/'actor_metadata.json').read_text())
    module=importlib.import_module('envs.pick_diverse_bottles')
    original=module.rand_create_actor
    def replacement(scene,modelname,xlim,ylim,**kwargs):
        if modelname!='001_bottle':return original(scene,modelname=modelname,xlim=xlim,ylim=ylim,**kwargs)
        pose=rand_pose(xlim=xlim,ylim=ylim,**{k:v for k,v in kwargs.items() if k in ('zlim','ylim_prop','rotate_rand','rotate_lim','qpos')})
        raw_scene,pose=preprocess(scene,pose);builder=raw_scene.create_actor_builder()
        for hull in manifest['hulls']:builder.add_convex_collision_from_file(filename=hull,material=raw_scene.default_physical_material)
        builder.add_visual_from_file(filename=manifest['visual'])
        entity=builder.build(name='cwmj_wine_bottle');entity.set_pose(pose)
        actor=Actor(entity,metadata,mass=manifest['mass_kg'])
        bodies=[c for c in entity.get_components() if isinstance(c,physx.PhysxRigidDynamicComponent)]
        if len(bodies)!=1:raise ValueError('single dynamic body required')
        body=bodies[0];body.mass=manifest['mass_kg'];body.inertia=manifest['principal_inertia_kg_m2']
        body.cmass_local_pose=sapien.Pose(manifest['center_of_mass_m'],manifest['inertia_frame_quaternion_wxyz'])
        return actor
    module.rand_create_actor=replacement
    return {'manifest':str(ASSET/'manifest.json'),'manifest_sha256':sha256(ASSET/'manifest.json'),'original_requested_ids_are_rng_draws_only':True,'actual_geometry':'same_cwmj_wine_bottle_both_sides','mass_kg':manifest['mass_kg']}

"""Read-only CWMJ wine candidate -> isolated SAPIEN asset, explicit geometry map."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import trimesh
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/admin001/cwmj-horizontal/results/coacd_targeted/wine_bottle_main/visual_probe/candidate')
OUT=R/'work/cwmj_bottle_asset_v2'

def prepare():
    if OUT.exists():raise FileExistsError('never replace converted asset')
    manifest=json.loads((SOURCE/'candidate_manifest.json').read_text())
    xml=SOURCE/manifest['candidate_mjcf']
    if sha256(xml)!=manifest['candidate_mjcf_sha256']:raise ValueError('CWMJ XML hash mismatch')
    for part in manifest['parts']:
        if sha256(SOURCE/part['file'])!=part['sha256']:raise ValueError('CWMJ hull hash mismatch')
    tree=ET.parse(xml);body=tree.find(".//body[@name='object']");inertial=body.find('inertial')
    original_template=R/'work/overlay_policy_equivalent_pilot_v1/assets/objects/001_bottle/model_data14.json'
    template=json.loads(original_template.read_text())
    # Match the height of original template14, preserving the new wine shape.
    # CWMJ visual OBJ uses the XML's 0.5/0.505 scale. Physical hulls already do.
    assets={m.attrib['name']:m for m in tree.findall('./asset/mesh')}
    visual=[];orig_bounds=[];sources={str(xml):sha256(xml),str(SOURCE/'candidate_manifest.json'):sha256(SOURCE/'candidate_manifest.json'),str(original_template):sha256(original_template)}
    for name in ['wine_bottle_red_main','wine_bottle_red_logo','wine_bottle_red_cork']:
        path=SOURCE/(name+'.obj');mesh=trimesh.load(path,force='mesh',process=False)
        mesh.apply_scale(np.fromstring(assets[name+'_vis'].get('scale'),sep=' '))
        visual.append(mesh);orig_bounds.append(mesh.bounds)
    bounds=np.stack(orig_bounds);source_height=float(bounds[:,1,2].max()-bounds[:,0,2].min())
    target_height=float(template['extents'][1]*template['scale'][1]);factor=target_height/source_height
    rotate=np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=float)
    transform=np.eye(4);transform[:3,:3]=rotate*factor
    OUT.mkdir();(OUT/'collision').mkdir()
    for mesh in visual:mesh.apply_transform(transform)
    scene=trimesh.Scene(visual);scene.export(str(OUT/'visual.glb'))
    hulls=[]
    for i,part in enumerate(manifest['parts']):
        path=SOURCE/part['file'];mesh=trimesh.load(path,force='mesh',process=False)
        if not mesh.is_watertight or not mesh.is_convex:raise ValueError('invalid convex hull')
        geom=body.find(f"geom[@mesh='{part['mesh']}']")
        if geom is None or geom.get('quat') is not None:raise ValueError('unexpected hull transform')
        mesh.apply_translation(np.fromstring(geom.get('pos','0 0 0'),sep=' '));mesh.apply_transform(transform)
        out=OUT/f'collision/part_{i:03d}.obj';mesh.export(out);hulls.append(str(out))
    # Convert original grasp/functional coordinates from scaled template into m.
    metadata=dict(template)
    for key in ['contact_points_pose','target_pose','functional_matrix','orientation_point','transform_matrix']:
        if key in metadata:
            arr=np.asarray(metadata[key],dtype=float);arr[...,:3,3]*=np.asarray(template['scale']);metadata[key]=arr.tolist()
    metadata['scale']=[1.,1.,1.]
    metadata['center']=scene.bounds.mean(axis=0).tolist()
    metadata['extents']=np.ptp(scene.bounds,axis=0).tolist()
    (OUT/'actor_metadata.json').write_text(json.dumps(metadata,indent=2))
    # Preserve source density under uniform geometric scaling. Inertia frame is
    # rotated along with the mesh; diagonal eigenvalues scale by factor**5.
    from transforms3d.quaternions import quat2mat,mat2quat
    mass=float(inertial.get('mass'))*factor**3
    inertia=np.fromstring(inertial.get('diaginertia'),sep=' ')*factor**5
    com=rotate@np.fromstring(inertial.get('pos'),sep=' ')*factor
    inertia_q=mat2quat(rotate@quat2mat(np.fromstring(inertial.get('quat'),sep=' ')))
    for p in SOURCE.rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.obj','.mtl','.png','.jpg','.xml','.json']:sources[str(p)]=sha256(p)
    result={'schema':'lingbot_cwmj_sapien_bottle_candidate_v1','source_files_sha256':sources,
        'source_candidate_manifest_sha256':sha256(SOURCE/'candidate_manifest.json'),
        'source_height_m':source_height,'target_template_height_m':target_height,'uniform_scale':factor,
        'coordinate_map':'CWMJ Z-up -> RoboTwin Y-up: (x,y,z)->(x,z,-y); applied once to all visuals/hulls',
        'mass_kg':mass,'principal_inertia_kg_m2':inertia.tolist(),'center_of_mass_m':com.tolist(),'inertia_frame_quaternion_wxyz':inertia_q.tolist(),
        'hulls':hulls,'visual':str(OUT/'visual.glb'),'actor_metadata':str(OUT/'actor_metadata.json'),
        'limitations':['MuJoCo contact material is not equivalent to PhysX; use explicit local test material','visual OBJ variants used because candidate XML references absent .msh files; no bitwise render equivalence claim','grasp/functional metadata derived from original template14; manipulation not yet verified','single wine geometry; not original diverse 20-bottle distribution'],
        'training_admitted':False,'converted_files_sha256':{str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file()}}
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['source_height_m','target_template_height_m','uniform_scale','mass_kg','principal_inertia_kg_m2']}))
    return result
if __name__=='__main__':prepare()

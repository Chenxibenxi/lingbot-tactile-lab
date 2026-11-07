"""Fail-closed independent spatial contact sidecar auditor for FUTURE relay trials.

Existing seed100000 source has no such sidecar and MUST be rejected. This audit
can establish same-step projected 2D candidates, not actual RGB visibility,
calibrated grasp force, expert training approval, or a usable tactile model.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import h5py
from project_contact_geometry import CAMERAS,calibration_audit,project_world_contact
from validate_expert_data import read_jsonl,sha256
from validate_controlled_negative import build_card as negative_card
from register_dataset import build_card as expert_card


def validate_pair(raw,sidecar):
    rr=read_jsonl(raw);gg=read_jsonl(sidecar)
    if len(rr)<2 or len(rr)!=len(gg):
        raise ValueError('geometry/physics step count mismatch')
    rh,gh=rr[0],gg[0]
    if rh.get('schema')!='phizju_force_probe_v1' or gh.get('schema')!='lingbot_private_contact_geometry_v1':
        raise ValueError('unknown raw/geometry schemas')
    if gh.get('probe_file')!=Path(raw).name:
        raise ValueError('geometry references another physics stream')
    for field in ('task','seed','episode','clock_origin'):
        if gh.get(field)!=rh.get(field):
            raise ValueError(f'geometry {field} differs from physics')
    dt=rh.get('dt_s');gdt=gh.get('dt_s')
    if (type(dt) not in (float,int) or type(gdt) not in (float,int) or
            not math.isfinite(dt) or not math.isfinite(gdt) or dt<=0 or
            not math.isclose(dt,gdt,rel_tol=0.,abs_tol=1e-9)):
        raise ValueError('geometry physical dt mismatch')
    links=rh.get('side_links',{})
    if set(links)!={'left','right'} or any(not isinstance(links[s],list) or not links[s] for s in links):
        raise ValueError('raw finger link map missing')
    step_points=[];side_totals={'left':0,'right':0}
    for n,(r,g) in enumerate(zip(rr[1:],gg[1:]),1):
        if type(r.get('physics_step'))!=int or r['physics_step']!=n or type(g.get('physics_step'))!=int or g['physics_step']!=n:
            raise ValueError('geometry/physics step gap or duplicate')
        points=g.get('points')
        if not isinstance(points,list) or len(points)>1024:
            raise ValueError('invalid spatial point list')
        by_finger={(s,f):[0,0.] for s in ('left','right') for f in links[s]}
        for p in points:
            if not isinstance(p,dict) or set(p)!={'side','finger','position_world_m','load_proxy_n'}:
                raise ValueError('malformed/private point position payload')
            key=(p['side'],p['finger'])
            if key not in by_finger:raise ValueError('unknown spatial finger or side')
            xyz=p['position_world_m'];v=p['load_proxy_n']
            if not isinstance(xyz,list) or len(xyz)!=3 or any(type(x) not in (float,int) or not math.isfinite(x) for x in xyz):
                raise ValueError('nonfinite spatial world point')
            if type(v) not in (float,int) or not math.isfinite(v) or v<=0:
                raise ValueError('invalid nonzero spatial load proxy')
            by_finger[key][0]+=1;by_finger[key][1]+=v
        for side in ('left','right'):
            by_side=r.get(side,{});fingers=by_side.get('fingers',{})
            if set(fingers)!=set(links[side]):raise ValueError('raw finger map changed')
            count=0;load=0.
            for f in links[side]:
                recorded=fingers[f];num,sumproxy=by_finger[(side,f)]
                observed=recorded.get('points');value=recorded.get('load_proxy_n')
                if type(observed)!=int or observed<0 or observed!=num:
                    raise ValueError('spatial point count disagrees with finger raw')
                if type(value) not in (int,float) or not math.isfinite(value) or not math.isclose(value,sumproxy,rel_tol=1e-6,abs_tol=1e-10):
                    raise ValueError('spatial proxy disagrees with finger raw')
                count+=num;load+=sumproxy
            if type(by_side.get('points'))!=int or by_side['points']!=count or not math.isclose(float(by_side.get('load_proxy_n',float('nan'))),load,rel_tol=1e-6,abs_tol=1e-10):
                raise ValueError('spatial side total differs from raw')
            side_totals[side]+=count
        step_points.append(points)
    return {'step_points':step_points,'steps':len(step_points),'point_counts':side_totals,
            'header':rh}


def build_card(repo,trial):
    repo=Path(repo).resolve(strict=True);trial=Path(trial).resolve(strict=True)
    if trial.parent==repo/'data/negative':
        parent=negative_card(repo,trial)
        parent_path=repo/'datasets/negative_diagnostics'/f'{trial.name}.json'
        role='controlled_negative'
    elif trial.parent==repo/'data/paired':
        parent=expert_card(trial,repo)
        parent_path=repo/'datasets/cards'/f'{trial.name}.json'
        role='expert_candidate'
    else:raise ValueError('only relay paired or quarantined negative trial may contain spatial data')
    if parent!=json.loads(parent_path.read_text()):
        raise ValueError('parent card differs from validated same-run sources')
    if parent.get('source_machine_role')!='relay_single_rtx3090':
        raise ValueError('spatial contact is restricted to verified single-3090 relay provenance')
    raws=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl'))]
    if len(raws)!=1:raise ValueError('ambiguous raw source stream')
    raw=raws[0]
    folder=trial/'geometry'
    if not folder.is_dir() or folder.resolve(strict=True)!=folder:
        raise ValueError('missing private same-trial geometry directory')
    sidecar=folder/f'{raw.stem}.geometry.jsonl'
    if not sidecar.is_file() or sidecar.is_symlink():
        raise ValueError('missing matching geometry sidecar')
    pair=validate_pair(raw,sidecar)
    if (parent.get('task'),parent.get('seed'),parent.get('episode'))!=(pair['header']['task'],pair['header']['seed'],pair['header']['episode']):
        raise ValueError('spatial stream differs from verified trial identity')
    hdf5=trial/'demonstration/data'/f"episode{pair['header']['episode']}.hdf5"
    calibration=calibration_audit(hdf5)
    _,*frames=read_jsonl(raw.with_suffix('.frames.jsonl'))
    projections={cam:0 for cam in CAMERAS}
    with h5py.File(hdf5,'r') as f:
        if any(calibration['camera_calibration'][cam]['frames_with_calibration']!=len(frames) for cam in CAMERAS):
            raise ValueError('camera calibration not aligned to saved frame markers')
        for index,frame in enumerate(frames):
            step=frame['physics_step']
            if type(step)!=int or not 0<=step<=pair['steps']:
                raise ValueError('saved frame beyond spatial physical time')
            for cam in CAMERAS:
                prefix=f'observation/{cam}/'
                K=f[prefix+'intrinsic_cv'][index];E=f[prefix+'extrinsic_cv'][index]
                # Only actual SAME physical step points, not a history moved onto a later camera pose.
                for point in pair['step_points'][step-1] if step else ():
                    if project_world_contact(point['position_world_m'],K,E,(240,320)) is not None:
                        projections[cam]+=1
    return {'schema':'lingbot_spatial_contact_diagnostic_v1',
            'status':'private_same_run_frustum_projection_only_occlusion_unknown_not_training_or_eval',
            'task':parent['task'],'seed':parent['seed'],'episode':parent['episode'],
            'source_role':role,'source_machine_role':'relay_single_rtx3090',
            'physics_steps':pair['steps'],'camera_frames':len(frames),
            'world_contact_points_by_side':pair['point_counts'],
            'same_step_in_frustum_candidates_by_camera':projections,
            'geometry_sha256':sha256(sidecar),'parent_card_sha256':sha256(parent_path),
            'raw_sha256':sha256(raw),'hdf5_sha256':sha256(hdf5),
            'accepted_for_expert_training':False,'occlusion_validated':False,
            'caution':'2D frustum projections are not visible finger pixels or calibrated grip forces; no model observations contain world contact positions.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('refuse to overwrite card')
    if a.out.resolve().parent != (a.repo.resolve()/'datasets/spatial_diagnostics'):
        p.error('card must be under datasets/spatial_diagnostics')
    card=build_card(a.repo,a.trial);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as file:json.dump(card,file,indent=2);file.write('\n')
    print('SPATIAL_CONTACT_DIAGNOSTIC',card['source_role'],card['world_contact_points_by_side'],card['same_step_in_frustum_candidates_by_camera'])


if __name__=='__main__':main()

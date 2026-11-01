"""Private CPU replay: SAME-STEP world-contact projections, occlusion UNKNOWN.

The positive comparison supplies finger-link color badges only (no world points
were recorded in its old run). The new negative may additionally show outlined
2D projections of real SAPIEN point.position; neither is a skin stress sensor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from register_dataset import build_card as expert_card
from validate_controlled_negative import build_card as negative_card
from validate_contact_geometry import build_card as spatial_card,validate_pair
from project_contact_geometry import project_world_contact
from validate_expert_data import sha256
from render_hand_force_video import load_role,images_at,put_role,static_canvas,color_proxy,FONT,COLOR_TICKS

POSITIVE='relay_pickdiverse_s100007_expert0'
NEGATIVE='relay_pickdiverse_s100007_open_grasp_spatial1'


def overlay_same_step_points(canvas,source,idx,points,*,x=800):
    """Draw hollow projected point rings only from the displayed capture step."""
    cap=source['steps'][idx]
    if cap==0:return 0
    current=points[cap-1]
    count=0
    for p in current:
        for cam,left,top,width,height in (('head_camera',x+8,102,480,340),
                                          ('left_camera',x+488,102,310,165),
                                          ('right_camera',x+488,277,310,165)):
            if cam=='left_camera' and p['side']!='left' or cam=='right_camera' and p['side']!='right':
                continue
            prefix=f'observation/{cam}/'
            K=source['hdf5'][prefix+'intrinsic_cv'][idx];E=source['hdf5'][prefix+'extrinsic_cv'][idx]
            projected=project_world_contact(p['position_world_m'],K,E,(240,320))
            if projected is None:continue
            u,v,_=projected;cx=int(u*width/320);cy=int(v*height/240)
            # Draw into the camera ROI, not the full canvas: an edge contact
            # may have its ring CLIPPED, but must not spill into the title or
            # be displaced into a fabricated in-bounds hand pixel.
            view=canvas[top:top+height,left:left+width]
            cv2.circle(view,(cx,cy),9,(255,255,255),3,cv2.LINE_AA)
            cv2.circle(view,(cx,cy),7,color_proxy(p['load_proxy_n']),2,cv2.LINE_AA)
            count+=1
    return count


def spatial_canvas():
    frame=static_canvas()
    canvas=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT,16)
    draw.rectangle((0,883,1599,919),fill=(20,23,30))
    draw.text((12,888),'右侧白边色环=该图像拍照物理步真实三维触点的二维投影；遮挡未知，不等于画面可见受力/硬件夹力。左侧旧专家无三维点。',
              font=font,fill=(250,207,136))
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)


def render(repo,out,*,fps=25):
    repo=Path(repo).resolve(strict=True);out=Path(out)
    if out.exists() or fps not in (25,30):raise ValueError('fresh video path and 25/30fps only')
    if repo/'data/reports' not in out.resolve().parents:raise ValueError('private local data/reports only')
    pos=repo/'data/paired'/POSITIVE;neg=repo/'data/negative'/NEGATIVE
    pcard=expert_card(pos,repo);ncard=negative_card(repo,neg);scard=spatial_card(repo,neg)
    for path,card in ((repo/'datasets/cards'/f'{POSITIVE}.json',pcard),
                      (repo/'datasets/negative_diagnostics'/f'{NEGATIVE}.json',ncard),
                      (repo/'datasets/spatial_diagnostics'/f'{NEGATIVE}.json',scard)):
        if card!=json.loads(path.read_text()):raise ValueError('parent/spatial source card differs from live bytes')
    if (pcard['task'],pcard['seed'])!=(ncard['task'],ncard['seed']) or not ncard['goal_ok'] is False:
        raise ValueError('not a matched positive versus controlled failure')
    out.parent.mkdir(parents=True,exist_ok=True)
    positive=load_role(repo,'paired',POSITIVE);negative=load_role(repo,'negative',NEGATIVE)
    sources=(positive,negative)
    raw=next(p for p in (neg/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl')))
    geometry=neg/'geometry'/f'{raw.stem}.geometry.jsonl'
    geometry_points=validate_pair(raw,geometry)['step_points']
    for cam in ('head_camera','left_camera','right_camera'):
        if not np.array_equal(images_at(positive,0)[cam],images_at(negative,0)[cam]):
            raise ValueError('initial cameras not pixel-matched; refuse same-initial-state video')
    if not np.array_equal(positive['hdf5']['joint_action/vector'][0],negative['hdf5']['joint_action/vector'][0]):
        raise ValueError('initial 14D command not matched')
    ffmpeg=repo/'work/ffmpeg-bin/ffmpeg'
    if not ffmpeg.is_file():raise FileNotFoundError('private ffmpeg shim missing')
    duration=max(len(s['physics'])*s['dt'] for s in sources)
    count=math.ceil(duration*fps)+1;frame_base=spatial_canvas()
    command=[str(ffmpeg),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
             '-s','1600x920','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
             '-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    drawn=0;dot_frames=0
    try:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        for i in range(count):
            t=i/fps;frame=frame_base.copy()
            put_role(frame,0,positive,t,len(positive['physics'])*positive['dt'])
            idx=put_role(frame,800,negative,t,len(negative['physics'])*negative['dt'])
            # Link badges in the generic replay cover bottom wrist-camera pixels.
            # This spatial mode leaves both wrist images UNOBSTRUCTED; force colors
            # remain in the four per-finger bars below, and true dots stay on RGB.
            wrist=images_at(negative,idx)
            for cam,top in (('left_camera',102),('right_camera',277)):
                frame[top:top+165,1288:1598]=cv2.resize(wrist[cam],(310,165))
                cv2.putText(frame,cam.split('_')[0].upper()+' WRIST',(1296,top+25),
                            cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),2,cv2.LINE_AA)
            dots=overlay_same_step_points(frame,negative,idx,geometry_points)
            drawn+=dots;dot_frames+=int(dots>0)
            proc.stdin.write(frame.tobytes())
            if i in (0,count//2,count-1) or dots and dot_frames==1:
                cv2.imwrite(str(out.with_name(f'{out.stem}_preview_{i}.png')),frame)
        proc.stdin.close();stderr=proc.stderr.read(2000);code=proc.wait(timeout=90)
        if code!=0:raise RuntimeError('H264 encoder failed: '+stderr.decode(errors='replace')[:300])
    finally:
        for s in sources:s['hdf5'].close()
    if not out.is_file() or out.stat().st_size<1024:raise ValueError('encoded MP4 empty')
    info={'schema':'lingbot_spatial_contact_projection_video_v1',
          'status':'same_step_projected_candidates_occlusion_unknown_not_calibrated_force',
          'fps':fps,'frames':count,'duration_s':count/fps,
          'same_step_projection_rings_drawn_in_video':drawn,
          'video_frames_with_projection_rings':dot_frames,
          'positive_source_card_sha256':sha256(repo/'datasets/cards'/f'{POSITIVE}.json'),
          'negative_source_card_sha256':sha256(repo/'datasets/negative_diagnostics'/f'{NEGATIVE}.json'),
          'spatial_source_card_sha256':sha256(repo/'datasets/spatial_diagnostics'/f'{NEGATIVE}.json'),
          'video_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'color_reference_proxy':list(COLOR_TICKS),
          'no_visible_surface_or_hardware_force_claim':True}
    with out.with_suffix('.manifest.json').open('x') as f:json.dump(info,f,indent=2);f.write('\n')
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fps',type=int,default=25)
    a=p.parse_args();result=render(a.repo,a.out,fps=a.fps)
    print('SPATIAL_CONTACT_VIDEO',a.out,'rings',result['same_step_projection_rings_drawn_in_video'],
          'frames',result['frames'],'sha256',result['video_sha256'])


if __name__=='__main__':main()

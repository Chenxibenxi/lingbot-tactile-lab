"""CPU-only visual analysis of simulated finger contact proxy on SAME-RUN RGB.

Two synchronized seed100000 trials: unmodified expert vs opt-in open-grasp
failure. Color is log-scaled SAPIEN impulse/dt PROXY, not calibrated grip force.
The source lacks 3D contact-point camera projections: colored finger badges
are link-level annotations, NEVER exact pixel locations on the hands.
"""
from __future__ import annotations
import argparse
import bisect
import hashlib
import json
import math
import subprocess
from pathlib import Path
import cv2
import h5py
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from contact_persistence_audit import audit as audit_sources
from validate_expert_data import read_jsonl

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CAMERAS=('head_camera','left_camera','right_camera')
SIDES=('left','right')
LABELS=('L7','L8','R7','R8')
COLOR_TICKS=(0.,.05,.5,5.,50.)
BGR=((165,55,35),(195,175,35),(65,195,75),(35,225,220),(35,55,235))


def color_proxy(value):
    """Fixed shared log scale: blue zero; cyan .05; green .5; yellow 5; red 50+."""
    v=float(value)
    if not math.isfinite(v) or v<0:raise ValueError('invalid nonnegative contact proxy')
    if v>=50:return BGR[-1]
    transform=lambda x:math.log1p(x/.05)/math.log1p(50/.05)
    z=transform(v)
    stops=[transform(x) for x in COLOR_TICKS]
    idx=max(0,bisect.bisect_right(stops,z)-1)
    idx=min(idx,len(stops)-2)
    t=(z-stops[idx])/(stops[idx+1]-stops[idx])
    return tuple(int(round((1-t)*BGR[idx][c]+t*BGR[idx+1][c])) for c in range(3))


def load_role(root,subdir,trial):
    path=root/'data'/subdir/trial
    raws=[p for p in (path/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl'))]
    if len(raws)!=1:raise ValueError('ambiguous same-run ForceProbe')
    header,*physics=read_jsonl(raws[0]);_,*markers=read_jsonl(raws[0].with_suffix('.frames.jsonl'))
    if not physics or not markers or [row['physics_step'] for row in physics]!=list(range(1,len(physics)+1)):
        raise ValueError('missing physical clock in source')
    steps=[int(row['physics_step']) for row in markers]
    if steps!=sorted(steps):raise ValueError('nonmonotonic camera markers')
    order=[(side,finger) for side in SIDES for finger in sorted(header['side_links'][side])]
    if len(order)!=4:raise ValueError('expected exactly four known finger links')
    peak_by_frame=[];now_by_frame=[]
    for step in steps:
        rows=physics[max(0,step-25):step]   # physics row index step-1, <= camera capture
        peak_by_frame.append([max((float(r[s]['fingers'][f]['load_proxy_n']) for r in rows),default=0.)
                              for s,f in order])
        now_by_frame.append([float(physics[step-1][s]['fingers'][f]['load_proxy_n']) if step else 0.
                             for s,f in order])
    return {'path':path,'physics':physics,'steps':steps,'dt':float(header['dt_s']),
            'peak':peak_by_frame,'instant':now_by_frame,'order':order,
            'hdf5':h5py.File(path/'demonstration/data/episode0.hdf5','r'),
            'cached_index':None,'cached_images':None}


def images_at(source,idx):
    if source['cached_index']==idx:return source['cached_images']
    out={}
    for cam in CAMERAS:
        b=source['hdf5'][f'observation/{cam}/rgb'][idx]
        rgb=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape!=(240,320,3):raise ValueError('same-run RGB corrupt')
        out[cam]=rgb
    source['cached_index']=idx;source['cached_images']=out
    return out


def static_canvas():
    h,w=920,1600
    image=Image.new('RGB',(w,h),(24,27,35));draw=ImageDraw.Draw(image)
    title=ImageFont.truetype(FONT,29);label=ImageFont.truetype(FONT,21);small=ImageFont.truetype(FONT,17)
    draw.text((28,12),'双臂夹爪接触代理强度 · 同种子成功 vs 受控开爪失败',font=title,fill=(245,249,255))
    draw.text((28,51),'原始仿真冲量÷步长的相对颜色；不是标定夹持力，也不是精确像素触点定位',font=small,fill=(252,194,101))
    for x,role in ((0,'规则专家成功 / 闭合抓取'),(800,'受控失败 / 两次抓取保持张开')):
        draw.text((x+14,77),role,font=label,fill=(145,240,155) if x==0 else (255,162,122))
        draw.text((x+15,446),'当前图像/传感器均取自同一拍照物理步；以下数值为四指100ms窗内峰值',font=small,fill=(185,200,216))
        draw.text((x+15,768),'全程接触时间轴：绿=左指接触，橙=右指接触；白线=当前视频位置',font=small,fill=(185,200,216))
        draw.text((x+15,864),'固定共用对数色标：蓝0 → 青0.05 → 绿0.5 → 黄5 → 红≥50 (代理量)',font=small,fill=(207,213,221))
    draw.text((12,891),'25fps 仅是可视化播放帧率；图像按真实仿真步号重采样。彩标对应夹爪指连杆，不代表精确触点像素。',font=small,fill=(174,182,193))
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)


def put_role(canvas,x,source,video_t,final_t):
    step=min(len(source['physics']),int(video_t/source['dt']+1e-8))
    idx=max(0,bisect.bisect_right(source['steps'],step)-1)
    cap=source['steps'][idx]
    images=images_at(source,idx)
    canvas[102:442,x+8:x+488]=cv2.resize(images['head_camera'],(480,340))
    canvas[102:267,x+488:x+798]=cv2.resize(images['left_camera'],(310,165))
    canvas[277:442,x+488:x+798]=cv2.resize(images['right_camera'],(310,165))
    cv2.putText(canvas,'HEAD RGB',(x+18,132),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),2,cv2.LINE_AA)
    for side,y,indices in (('LEFT WRIST',267,(0,1)),('RIGHT WRIST',442,(2,3))):
        cv2.putText(canvas,side,(x+496,y-136),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),2,cv2.LINE_AA)
        for col,j in enumerate(indices):
            xx=x+495+col*151;yy=y-33
            colr=color_proxy(source['peak'][idx][j]);cv2.rectangle(canvas,(xx,yy),(xx+132,yy+26),colr,-1)
            cv2.putText(canvas,f'{LABELS[j]} {source["peak"][idx][j]:.2f}',(xx+5,yy+18),cv2.FONT_HERSHEY_SIMPLEX,.46,(8,8,8),1,cv2.LINE_AA)
    cv2.putText(canvas,f'image t={cap*source["dt"]:.3f}s  step={cap}  playback={video_t:.2f}s',
                (x+16,489),cv2.FONT_HERSHEY_SIMPLEX,.58,(239,244,247),1,cv2.LINE_AA)
    for j,tag in enumerate(LABELS):
        y=507+j*61;peak=source['peak'][idx][j];instant=source['instant'][idx][j]
        cv2.rectangle(canvas,(x+13,y),(x+787,y+51),(49,55,68),-1)
        cv2.putText(canvas,tag,(x+26,y+31),cv2.FONT_HERSHEY_SIMPLEX,.7,(244,248,252),2,cv2.LINE_AA)
        cv2.rectangle(canvas,(x+96,y+11),(x+456,y+37),(80,86,96),-1)
        ratio=math.log1p(min(50.,peak)/.05)/math.log1p(50/.05)
        if peak>0:cv2.rectangle(canvas,(x+96,y+11),(x+96+max(3,int(360*ratio)),y+37),color_proxy(peak),-1)
        cv2.putText(canvas,f'peak {peak:6.3f}  now {instant:6.3f}',(x+467,y+31),
                    cv2.FONT_HERSHEY_SIMPLEX,.57,(237,244,250),1,cv2.LINE_AA)
    # Full-run timeline is a POST-HOC visualization, not a model input.
    for side,y,col in (('left',807,(88,218,125)),('right',833,(80,160,245))):
        cv2.rectangle(canvas,(x+72,y),(x+745,y+12),(49,55,68),-1)
        for row in source['physics']:
            if row[side]['points']>0:
                xx=x+72+int(row['physics_step']/len(source['physics'])*673)
                cv2.line(canvas,(xx,y),(xx,y+12),col,1)
        cv2.putText(canvas,'L' if side=='left' else 'R',(x+35,y+11),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1)
        cursor=x+72+int(min(video_t,final_t)/final_t*673)
        cv2.line(canvas,(cursor,y-5),(cursor,y+17),(250,251,252),1)
    return idx


def render(repo,out,*,fps=25):
    repo=Path(repo).resolve(strict=True);out=Path(out)
    if out.exists() or fps not in (25,30):raise ValueError('new video path and supported fps required')
    if repo/'data/reports' not in out.resolve().parents:
        raise ValueError('video must remain local under ignored data/reports')
    evidence=audit_sources(repo)  # verifies BOTH source cards, HDF5s, initial RGB/state
    ffmpeg=repo/'work/ffmpeg-bin/ffmpeg'
    if not ffmpeg.is_file() or not ffmpeg.stat().st_mode & 0o111:raise FileNotFoundError('private ffmpeg shim missing')
    out.parent.mkdir(parents=True,exist_ok=True)
    positive=load_role(repo,'paired','relay_pickdiverse_s100000_expert0')
    negative=load_role(repo,'negative','relay_pickdiverse_s100000_open_grasp1')
    sources=(positive,negative)
    base=static_canvas();duration=max(len(s['physics'])*s['dt'] for s in sources)
    count=int(math.ceil(duration*fps))+1
    command=[str(ffmpeg),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
             '-s','1600x920','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
             '-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    try:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        for i in range(count):
            t=i/fps;frame=base.copy()
            for x,s in ((0,positive),(800,negative)):
                put_role(frame,x,s,t,len(s['physics'])*s['dt'])
            proc.stdin.write(frame.tobytes())
            if i in (0,count//2,count-1):
                cv2.imwrite(str(out.with_name(f'{out.stem}_preview_{i}.png')),frame)
        proc.stdin.close()
        stderr=proc.stderr.read(2000);code=proc.wait(timeout=90)
        if code!=0:raise RuntimeError('ffmpeg H264 failed: '+stderr.decode(errors='replace')[:300])
    finally:
        for s in sources:s['hdf5'].close()
    if not out.is_file() or out.stat().st_size<1024:raise ValueError('empty encoded video')
    info={'schema':'lingbot_hand_contact_proxy_video_v1','status':'local_analysis_only_not_calibrated_force',
          'fps':fps,'frames':count,'duration_visualization_s':count/fps,
          'positive_source_physics_steps':len(positive['physics']),
          'negative_source_physics_steps':len(negative['physics']),
          'source_positive_card_sha256':evidence['positive_card_sha256'],
          'source_negative_card_sha256':evidence['negative_card_sha256'],
          'color_reference_proxy':list(COLOR_TICKS),
          'video_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'camera_alignment':'held at latest same-run capture <= physical playback time',
          'finger_colors':'four link-level 100ms rolling peaks, NOT projected contact pixel positions'}
    with out.with_suffix('.manifest.json').open('x') as f:json.dump(info,f,indent=2);f.write('\n')
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--fps',type=int,default=25)
    args=p.parse_args();info=render(args.repo,args.out,fps=args.fps)
    print('HAND_FORCE_ANALYSIS_VIDEO',args.out,'frames',info['frames'],'sha256',info['video_sha256'])


if __name__=='__main__':main()

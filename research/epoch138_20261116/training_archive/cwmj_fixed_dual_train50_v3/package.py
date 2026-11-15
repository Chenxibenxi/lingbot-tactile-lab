"""Package exactly50 validated videos; technical teacher admission stays explicit."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
from PIL import Image,ImageDraw
R=Path('/mnt/relaydata/lingbot-tactile-lab');D=R/'data/cwmj_fixed_dual_train50_v3';O=R/'data/reports/cwmj_fixed_dual_train50_v3';W=R/'data/cwmj_fixed_dual_train50_windows_v3';W.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((O/'selected_training_sources.json').read_text());rows=[];windows=[];trajectories=[]
platform_sha=sha(R/'datasets/cwmj_fixed_dual_arm_platform_v2.json')
for case in plan['training']:
 d=Path(case['source_dir']);validation=json.loads((d/'validation.json').read_text());assert validation['task_success'] and validation['provenance']['status']=='passed'
 completion=json.loads((d/'completion.json').read_text());assert all(sha(d/name)==h for name,h in completion['files'].items())
 identity=json.loads((d/'identity.json').read_text());assert identity['platform_config_sha256']==platform_sha and identity['scenario']['xy_offset_m']==case['offset_m']
 result=json.loads((d/'result.json').read_text());records=result['records'];assert len(records)==163 and all(x['execution']['accepted'] for x in records)
 actions=np.asarray([x['execution']['actual']['raw_action']['action_14d_normalized'] for x in records],dtype=np.float32);assert actions.shape==(163,14) and np.isfinite(actions).all();assert ((actions[:,[6,13]]>=0)&(actions[:,[6,13]]<=1)).all()
 f=W/f"{case['seed']}_actions.npy";np.save(f,actions);trajectories.append({'seed':case['seed'],'active_side':case['side'],'file':f.name,'sha256':sha(f),'source_completion_sha256':sha(d/'completion.json')})
 for i,row in enumerate(records):
  p=d/row['pre']['metadata'];m=json.loads(p.read_text());assert f"Use the {case['side']} arm" in m['observation']['language']['instruction'];intervals=m['causal_tactile_intervals'];now=m['observation']['decision_time_s'];assert all(v['interval_end_s']<=now+1e-9 for v in intervals)
  windows.append({'seed':case['seed'],'scenario_id':case['case_id'],'split':'train','offset':i,'valid_steps':min(50,len(records)-i),'stage':row['stage'],'observation':str(p),'observation_sha256':sha(p),'causal_intervals':len(intervals)})
 rows.append({**case,'video':str((d/'rollout.mp4').resolve()),'video_sha256':validation['video_sha256'],'validation_sha256':sha(d/'validation.json'),'task_success':True})
assert len(rows)==50 and len(windows)==8150
for side in ['left','right']:
 group=[r for r in rows if r['side']==side];assert len(group)==25 and len({tuple(r['offset_m']) for r in group})==25
 for kind,frame in [('initial','reset_images.npz'),('grasp','observation_082_images.npz')]:
  sheet=Image.new('RGB',(5*320,5*264),'white');draw=ImageDraw.Draw(sheet)
  for j,row in enumerate(group):
   x,y=(j%5)*320,(j//5)*264
   with np.load(Path(row['source_dir'])/frame) as images:sheet.paste(Image.fromarray(images['cam_high']),(x,y+24))
   draw.text((x+4,y+5),f"{row['case_id']} xy={row['offset_m']}",fill='black')
  sheet.save(O/f'{side}_{kind}_sheet.jpg',quality=92)
 concat=O/f'{side}_videos.txt';concat.write_text(''.join("file '"+r['video'].replace("'","'\\''")+"'\n" for r in group))
 subprocess.run(['/home/admin001/miniconda3/bin/ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(O/f'{side}_all25_full.mp4')],check=True)
index={'schema':'cwmj_full_trajectory_windows_v1','horizon_steps':50,'action_step_s':.05,'training_admitted':False,'platform_config_sha256':platform_sha,'trajectories':trajectories,'windows':windows};(W/'index.json').write_text(json.dumps(index,indent=2))
admission={'training_admitted':False,'index_sha256':sha(W/'index.json'),'successful_videos':50,'left_videos':25,'right_videos':25,'human_authorized':True,'technical_teacher_pass':False,'remaining':'teacher semantic/causal admission evidence; report actual status, never infer from video success alone'};(W/'admission.json').write_text(json.dumps(admission,indent=2))
(O/'video_manifest.json').write_text(json.dumps({'count':50,'same_run':True,'fps':20,'playback_rate':1,'rows':rows},indent=2))
lines=['# 首任务：50条双臂场景抓放视频','', '左侧左臂25条、右侧右臂25条；固定30厘米腕间距初始姿态、相同相机和机器人。每条视频含重置至终止，20FPS、仿真1倍速。','', '完整任务质量和同源校验通过；技术教师准入独立记录，尚不能据此宣称模型具备能力。','']
for side in ['left','right']:
 lines.extend([f'## {side}:25条完整视频合集',f'![{side}合集]({(O/f"{side}_all25_full.mp4").resolve()})','',f'![{side}初始位置]({(O/f"{side}_initial_sheet.jpg").resolve()})',''])
 for row in [r for r in rows if r['side']==side]:lines.append(f"- [{row['case_id']}，偏移{row['offset_m']}米]({row['video']})")
(O/'index.md').write_text('\n'.join(lines)+'\n');print('packaged50 videos,8150 windows; teacher admission remains separate')

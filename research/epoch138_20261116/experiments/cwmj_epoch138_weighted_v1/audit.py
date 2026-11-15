from pathlib import Path
import json,hashlib,subprocess
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/mnt/relaydata/lingbot-tactile-lab');D=R/'data/cwmj_epoch138_weighted_v1';summary=json.loads((D/'summary.json').read_text());cache={};rows=[]
def read(p):return json.loads(p.read_text())
def bound(p,sha):
 if str(p) not in cache:cache[str(p)]=(hashlib.sha256(p.read_bytes()).hexdigest(),read(p))
 assert cache[str(p)][0]==sha;return cache[str(p)][1]
def rot(p):q=p['quaternion_wxyz'];return Rotation.from_quat(q[1:]+q[:1])
common={}
for e in summary['episodes']:
 base=e['case'].split('__')[0];common[base]=min(common.get(base,999),e['metrics']['executed_model_ticks'])
for ep in summary['episodes']:
 d=D/ep['case'];base,cfg=ep['case'].split('__');res=read(d/'result.json');records=res['records'][1:];side=ep['side'];arm=list(range(7,13)) if side=='right' else list(range(6));stateidx=list(range(6,12)) if side=='right' else list(range(6));grip=13 if side=='right' else 6;fingers=['fr_link7','fr_link8'] if side=='right' else ['fl_link7','fl_link8'];hand='fr_link6' if side=='right' else 'fl_link6';targets=[];tracking=[];samples=[]
 for r in records:
  en=r['admission']['ensemble'];src=en['sources'];tick=r['index'];ages=[tick-s['tick'] for s in src];assert ages==list(range(len(src)-1,-1,-1));a=[]
  for s,age in zip(src,ages):
   ans=bound(Path(s['response_file']),s['response_sha256']);bound(Path(s['metadata_file']),s['metadata_sha256']);dec=ans['decoded'];arr=np.array(dec.get('commands_14d',dec.get('raw_commands_14d')));assert age==s['source_row'] and tick==s['target_tick'];assert np.allclose(arr[age],s['raw_selected_row'],rtol=0,atol=0);a.append(arr[age])
  weights=np.power(en['config']['decay'],np.array(ages,dtype=float));weights/=sum(weights);assert np.allclose(weights,en['weights']);expect=np.array(a[-1]);ai=[0,1,2,3,4,5,7,8,9,10,11,12];expect[ai]=np.sum(np.array(a)[:,ai]*weights[:,None],axis=0);expect[[6,13]]=np.clip(expect[[6,13]],0,1);assert np.allclose(expect,r['admission']['command'])
  role=r['arm_role'];assert np.allclose(np.array(role['command_after_role'])[arm],expect[arm]);ex=r['execution']
  if not ex['accepted']:continue
  targets.append(role['command_after_role']);steps=ex['actual']['execution_substeps'];tracking.append(max(max(abs(s['reference_arm'][i]-s['arm_state'][i]['position_rad']) for i in stateidx) for s in steps));samples+=read(d/r['diagnostic_file'])
 origin=read(d/'origin.json')['post_integration_poses'];z0=origin['tomato_sauce']['position_m'][2];lift=[];contacts=[];relative=[]
 for s in samples:
  p=s['diagnostic_trace']['post_integration_poses'];o=p['tomato_sauce'];h=p[hand];lift.append((o['position_m'][2]-z0)*1000);contacts.append(s['sensor_valid'] and max(s['per_finger_max_normal_N'][f] for f in fingers)>.01);relative.append((rot(h).inv().apply(np.array(o['position_m'])-h['position_m']),rot(h).inv()*rot(o)))
 contactidx=np.flatnonzero(contacts);gap=run=0
 if len(contactidx):
  for b in contacts[contactidx[0]:contactidx[-1]+1]:run=0 if b else run+1;gap=max(gap,run)
 elevated=[i for i in range(len(samples)) if lift[i]>=30 and contacts[i]];relmm=relrot=None
 if elevated:
  t0,q0=relative[elevated[0]];relmm=max(np.linalg.norm(relative[i][0]-t0)*1000 for i in elevated);relrot=max((q0.inv()*relative[i][1]).magnitude()*180/np.pi for i in elevated)
 t=np.array(targets);n=common[base];jumps=np.max(np.abs(np.diff(t[:n,arm],axis=0)),axis=1)*180/np.pi;g=np.diff(t[:n,grip])*.055;rev=int(np.sum((g[1:]*g[:-1]<0)&(np.abs(g[1:])>.0001)&(np.abs(g[:-1])>.0001)))
 c=read(d/'completion.json');f=read(d/'force_review_data.json');assert hashlib.sha256((d/'rollout.mp4').read_bytes()).hexdigest()==c['video_sha256']==f['video_sha256'];assert len(f['frames'])==c['frames'];assert all(len(x['samples'])==25 for x in f['frames'][1:]);subprocess.run(['/home/admin001/miniconda3/bin/ffmpeg','-v','error','-i',str(d/'rollout.mp4'),'-f','null','-'],check=True)
 row={'case':base,'config':cfg,'metrics':ep['metrics'],'termination':ep['termination'],'stop':records[-1]['execution'].get('preflight',{}).get('stop'),'max_lift_mm':max(lift,default=0),'max_both_no_force_gap_between_first_last_contact_ms':gap*2 if len(contactidx) else None,'common_prefix_ticks':n,'common_prefix_target_jump_median_deg':float(np.median(jumps)),'common_prefix_tracking_max_deg':float(max(tracking[:n])*180/np.pi),'common_prefix_gripper_reversals':rev,'elevated_contact_relative_translation_range_from_first_mm':relmm,'elevated_contact_relative_rotation_from_first_deg':relrot};rows.append(row);print(cfg,base,round(row['max_lift_mm'],2),row['stop'],row['max_both_no_force_gap_between_first_last_contact_ms'],flush=True)
(D/'comparison_audit.json').write_text(json.dumps({'rows':rows,'causal_ensemble_and_video_checks':True,'scope':'28 reused development cases; no best-generalization claim. Force gap: both active fingers <=.01N between first and last >.01N contact, may include release/regrasp; not proof of lost hold. Relative motion only at lift>=30mm and force contact, not pure slip; disconnected spans included. Prefix metrics aligned per case across all configs.'},indent=2))

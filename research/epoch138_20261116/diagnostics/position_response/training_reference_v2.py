from pathlib import Path
import json,hashlib,numpy as np,mujoco
from cwmj_fixed_dual_arm_platform import FixedDualArmScenarioEnv
R=Path('/mnt/relaydata/lingbot-tactile-lab');W=Path(__file__).parent;D=R/'data/cwmj_fixed_dual_eval138_v2';plan=json.loads((R/'work/cwmj_fixed_dual_eval138_v2/plan.json').read_text());rows=[];I=R/'data/cwmj_fixed_dual_train50_windows_v3';index=json.loads((I/'index.json').read_text());catalog=json.loads((R/'datasets/cwmj_fixed_dual_position_generalization_v1.json').read_text());cases={x['seed']:x for x in catalog['training']};windows={x['seed']:x for x in index['windows'] if x['offset']==0}
for tr in index['trajectories']:
 d=Path(windows[tr['seed']]['observation']).parent;c=json.loads((d/'result.json').read_text())['position_case'];assert c['seed']==tr['seed'];key=c['case_id'];f=I/tr['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==tr['sha256'];a=np.load(f);assert a.shape==(163,14);records=json.loads((d/'result.json').read_text())['records']
 env=FixedDualArmScenarioEnv({'namespace':'fixed_dual_reserved_test','seed':c['seed'],'xy_offset_m':c['offset_m'],'yaw_offset_rad':0},active_side=c['side'],render=False,platform_path=R/plan['platform']);m=env.model;prefix='fl' if c['side']=='left' else 'fr';oi=0 if c['side']=='left' else 7;ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{prefix}_joint{i}') for i in range(1,9)];target=json.loads((d/'origin.json').read_text())['post_integration_poses']['tomato_sauce']['position_m'];pred=[]
 for k in [0,9,19,29,49]:
  fk=mujoco.MjData(m);mujoco.mj_copyData(fk,m,env.data);q=a[k,oi:oi+6];bounded=all(m.jnt_range[j][0]<=v<=m.jnt_range[j][1] for j,v in zip(ids[:6],q))
  for j,v in zip(ids[:6],q):fk.qpos[m.jnt_qposadr[j]]=v
  grip=float(-.01+.055*np.clip(a[k,oi+6],0,1))
  for j in ids[6:]:fk.qpos[m.jnt_qposadr[j]]=grip
  mujoco.mj_forward(m,fk);pts=[fk.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,prefix+f'_link{i}')].copy() for i in [7,8]];mid=np.mean(pts,axis=0)
  pred.append({'source_row':k,'teacher_stage':records[k]['stage'],'arm_in_bounds':bounded,'finger_body_midpoint':mid.tolist(),'xy_error_to_initial_object_mm':((mid[:2]-target[:2])*1000).tolist(),'gripper_normalized_raw':float(a[k,oi+6])})
 rows.append({'case':key,'side':c['side'],'offset_m':c['offset_m'],'initial_object_position_offline_only':target,'response_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'predictions':pred});env.close();print(key,flush=True)
summary=[]
for side in ['left','right']:
 pool=[x for x in rows if x['side']==side];truth=np.array([x['initial_object_position_offline_only'][:2] for x in pool]);X=np.c_[truth-truth.mean(axis=0),np.ones(len(pool))]
 for idx,k in enumerate([0,9,19,29,49]):
  est=np.array([x['predictions'][idx]['finger_body_midpoint'][:2] for x in pool]);coef=np.linalg.lstsq(X,est-est.mean(axis=0),rcond=None)[0];err=est-truth;summary.append({'side':side,'row':k,'linear_position_response_2x2':coef[:2].tolist(),'mean_xy_error_mm':(err.mean(axis=0)*1000).tolist(),'xy_rmse_mm':float(np.sqrt(np.mean(np.sum(err**2,axis=1)))*1000),'predicted_xy_span_mm':(np.ptp(est,axis=0)*1000).tolist(),'object_xy_span_mm':(np.ptp(truth,axis=0)*1000).tolist()})
(W/'training_reference.json').write_text(json.dumps({'rows':rows,'summary':summary,'scope':'Offline FK of actual50 training target sequences, verified index action SHA, matched row indices; teacher phases recorded. No physics rollout or new inference. Body-origin midpoint is not fingertip contact surface/grasp center; frozen initial object comparison ignores future motion; late predicted row is not necessarily a grasp phase. Regression sensitivity is not proof of visual grounding or success. All labels offline only.'},indent=2))

from pathlib import Path
import json,hashlib,numpy as np,mujoco
from cwmj_fixed_dual_arm_platform import FixedDualArmScenarioEnv
R=Path('/mnt/relaydata/lingbot-tactile-lab');W=Path(__file__).parent;D=R/'data/cwmj_fixed_dual_eval138_v2';plan=json.loads((R/'work/cwmj_fixed_dual_eval138_v2/plan.json').read_text());rows=[]
for c in plan['episodes_order']:
 if c['repeat']!=0:continue
 key=c['case_id']+'_r0';d=D/key;r=json.loads((d/'result.json').read_text())['records'][1];f=Path(r['response_file']);ans=json.loads(f.read_text());dec=ans['decoded'];a=np.asarray(dec.get('commands_14d',dec.get('raw_commands_14d')));assert a.shape==(50,14)
 env=FixedDualArmScenarioEnv({'namespace':'fixed_dual_reserved_test','seed':c['seed'],'xy_offset_m':c['offset_m'],'yaw_offset_rad':0},active_side=c['side'],render=False,platform_path=R/plan['platform']);m=env.model;prefix='fl' if c['side']=='left' else 'fr';oi=0 if c['side']=='left' else 7;ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{prefix}_joint{i}') for i in range(1,9)];target=json.loads((d/'origin.json').read_text())['post_integration_poses']['tomato_sauce']['position_m'];pred=[]
 for k in [0,9,19,29,49]:
  fk=mujoco.MjData(m);mujoco.mj_copyData(fk,m,env.data);q=a[k,oi:oi+6];bounded=all(m.jnt_range[j][0]<=v<=m.jnt_range[j][1] for j,v in zip(ids[:6],q))
  for j,v in zip(ids[:6],q):fk.qpos[m.jnt_qposadr[j]]=v
  grip=float(-.01+.055*np.clip(a[k,oi+6],0,1))
  for j in ids[6:]:fk.qpos[m.jnt_qposadr[j]]=grip
  mujoco.mj_forward(m,fk);pts=[fk.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,prefix+f'_link{i}')].copy() for i in [7,8]];mid=np.mean(pts,axis=0)
  pred.append({'source_row':k,'arm_in_bounds':bounded,'finger_body_midpoint':mid.tolist(),'xy_error_to_initial_object_mm':((mid[:2]-target[:2])*1000).tolist(),'gripper_normalized_raw':float(a[k,oi+6])})
 rows.append({'case':key,'side':c['side'],'offset_m':c['offset_m'],'initial_object_position_offline_only':target,'response_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'predictions':pred});env.close();print(key,flush=True)
summary=[]
for side in ['left','right']:
 pool=[x for x in rows if x['side']==side];truth=np.array([x['initial_object_position_offline_only'][:2] for x in pool]);X=np.c_[truth-truth.mean(axis=0),np.ones(len(pool))]
 for idx,k in enumerate([0,9,19,29,49]):
  est=np.array([x['predictions'][idx]['finger_body_midpoint'][:2] for x in pool]);coef=np.linalg.lstsq(X,est-est.mean(axis=0),rcond=None)[0];err=est-truth;summary.append({'side':side,'row':k,'linear_position_response_2x2':coef[:2].tolist(),'mean_xy_error_mm':(err.mean(axis=0)*1000).tolist(),'xy_rmse_mm':float(np.sqrt(np.mean(np.sum(err**2,axis=1)))*1000),'predicted_xy_span_mm':(np.ptp(est,axis=0)*1000).tolist(),'object_xy_span_mm':(np.ptp(truth,axis=0)*1000).tolist()})
(W/'summary.json').write_text(json.dumps({'rows':rows,'summary':summary,'scope':'Offline FK of first-observation predicted chunks from32 historical test scenes, repeat0. No physics rollout or new inference. Body-origin midpoint is not fingertip contact surface/grasp center; frozen initial object comparison ignores future motion; late predicted row is not necessarily a grasp phase. Regression sensitivity is not proof of visual grounding or success. All labels offline only.'},indent=2))

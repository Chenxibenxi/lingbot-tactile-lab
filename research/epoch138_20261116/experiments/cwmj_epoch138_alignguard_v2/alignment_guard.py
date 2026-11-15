"""Experimental robot-only guard; no object pose or pixel ground truth inputs."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
class AlignmentGuard:
 def __init__(self,model,side):
  self.m=model;self.side=side;self.contacted=False;self.no_contact_ticks=0;self.phase_start=0;self.rearms=0;self.prefix='fl' if side=='left' else 'fr';self.offset=0 if side=='left' else 7
  names=[f'{p}_joint{i}' for p in ['fl','fr'] for i in range(1,8)];self.ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];self.qadr=[model.jnt_qposadr[j] for j in self.ids];self.dof=[model.jnt_dofadr[j] for j in self.ids[self.offset:self.offset+6]];self.armadr=self.qadr[self.offset:self.offset+6];self.body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,self.prefix+'_link6')
 def fk(self,measured,arm=None):
  d=mujoco.MjData(self.m);d.qpos[:]=self.m.qpos0
  d.qpos[self.qadr]=measured
  if arm is not None:d.qpos[self.armadr]=arm
  mujoco.mj_forward(self.m,d);return d,d.xpos[self.body].copy(),d.xmat[self.body].reshape(3,3).copy()
 def apply(self,command,measured,force,tick):
  audit={'scope':'experimental robot-only alignment guard; not object localization','tick':tick,'input_command':list(command),'measured_robot_q14':list(measured)}
  fs=force.get('per_finger',{})
  valid=force.get('sensor_valid') is True and all(fs.get(self.prefix+f'_link{i}',{}).get('max_normal_N') is not None for i in [7,8])
  contact=valid and any(fs[self.prefix+f'_link{i}']['max_normal_N']>.01 for i in [7,8])
  if contact:self.contacted=True;self.no_contact_ticks=0
  elif valid:self.no_contact_ticks+=1
  else:self.no_contact_ticks=0
  if self.contacted and self.no_contact_ticks>=5:
   self.contacted=False;self.phase_start=tick;self.rearms+=1
  audit.update(contact_latched=self.contacted,no_contact_ticks=self.no_contact_ticks,phase_start=self.phase_start,rearm_count=self.rearms,valid_force=valid)
  if self.contacted:return list(command),dict(audit,applied=False)
  current,p,R=self.fk(measured);target,g,G=self.fk(measured,command[self.offset:self.offset+6]);err=float(np.linalg.norm(g[:2]-p[:2]));audit.update(current_position=p.tolist(),model_target_position=g.tolist(),lateral_error_m=err)
  if err<=.01 or g[2]>=p[2]-.001:return list(command),dict(audit,applied=False)
  if tick-self.phase_start>=40 or self.rearms>2:return None,dict(audit,applied=False,rejected='alignment_budget_exhausted')
  goal=g.copy();goal[2]=p[2]-.001;q=np.asarray(command[self.offset:self.offset+6],float).copy();jp=np.zeros((3,self.m.nv));jr=jp.copy()
  for it in range(60):
   d,x,Q=self.fk(measured,q);dp=goal-x;dr=Rotation.from_matrix(G@Q.T).as_rotvec()
   if np.linalg.norm(dp)<=.0005 and np.linalg.norm(dr)<=.005:break
   mujoco.mj_jacBody(self.m,d,jp,jr,self.body);J=np.vstack([jp[:,self.dof],jr[:,self.dof]]);delta=J.T@np.linalg.solve(J@J.T+.0001*np.eye(6),np.r_[dp,dr]);q+=np.clip(delta,-.1,.1)
  else:return None,dict(audit,applied=False,rejected='ik_not_converged')
  limits=self.m.jnt_range[self.ids[self.offset:self.offset+6]]
  if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):return None,dict(audit,applied=False,rejected='ik_joint_bounds')
  out=list(command);out[self.offset:self.offset+6]=q.tolist();return out,dict(audit,applied=True,projected_position=goal.tolist(),output_command=out,ik_position_error_m=float(np.linalg.norm(dp)),ik_rotation_error_rad=float(np.linalg.norm(dr)),iterations=it)

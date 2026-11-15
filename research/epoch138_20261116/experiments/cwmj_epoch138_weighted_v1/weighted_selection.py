"""Causal same-target-time ensemble; newest grippers, no arm clipping."""
import copy
import numpy as np
from mainline_gripper_projection import admit_projected_first_target
ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
def select_weighted(history,tick,config,validate):
 window=config['window'];decay=config['decay']
 if window not in (1,3,5,9) or not 0<decay<=1:raise ValueError('invalid bounded grid')
 if not history or history[-1]['tick']!=tick:raise ValueError('missing current prediction')
 used=history[-window:];ages=[tick-h['tick'] for h in used]
 if ages!=list(range(len(used)-1,-1,-1)):raise ValueError('noncausal or noncontiguous history')
 sources=[];commands=[];latest=None
 for h,age in zip(used,ages):
  d=copy.deepcopy(h['decoded']);key='commands_14d' if d.get('status')=='DECODED' else 'raw_commands_14d'
  a=np.asarray(d.get(key),dtype=float)
  if a.shape!=(50,14) or not np.isfinite(a).all():return {'accepted':False,'reason':'invalid_source_array'}
  d[key]=np.roll(a,-age,axis=0).tolist();ad=admit_projected_first_target(d,validate)
  if not ad['accepted']:return {'accepted':False,'reason':'source_row_rejected','source_tick':h['tick'],'source_admission':ad}
  commands.append(ad['command']);latest=ad
  sources.append({k:v for k,v in h.items() if k!='decoded'}|{'source_row':age,'target_tick':tick,'observation_age_s':age*.05,'raw_selected_row':a[age].tolist(),'admitted_selected_row':ad['command']})
 weights=np.power(decay,np.asarray(ages,dtype=float));weights/=weights.sum()
 command=np.array(commands[-1],dtype=float);command[ARM]=np.sum(np.asarray(commands)[:,ARM]*weights[:,None],axis=0)
 validated=validate(command.tolist(),gripper_units='normalized')
 result=copy.deepcopy(latest);result.update(command=command.tolist(),validated=validated,contract='same_target_time_weighted_arm_v1',ensemble={'config':config,'target_tick':tick,'sources':sources,'weights':weights.tolist(),'output':command.tolist(),'gripper_policy':'latest only','arm_clipping':False,'validated':validated})
 return result

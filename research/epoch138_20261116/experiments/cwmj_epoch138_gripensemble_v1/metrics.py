"""Full-task outcome audit; only executed physics, never rejected trial states."""
import json
from pathlib import Path
import numpy as np
from cwmj_slip_outcomes import relative

def evaluate_full_task(directory,origin,support_geom_ids,criteria,*,active_side):
 if active_side not in ("left","right"):raise ValueError("active side required")
 root=Path(directory);result=json.loads((root/'result.json').read_text());records=[v for v in result['records'] if v.get('stage')!='ready_warmup'];identity=json.loads((root/'identity.json').read_text());target=set(identity['target_geom_ids']);finger=set(sum(identity['finger_geom_ids'].values(),[]));support=set(support_geom_ids);samples=[]
 for row in records:
  if row.get('execution',{}).get('accepted'):samples.extend(json.loads((root/row['diagnostic_file']).read_text()))
 active_names=('fl_link7','fl_link8') if active_side=='left' else ('fr_link7','fr_link8');active_geoms=[set(identity['finger_geom_ids'][n]) for n in active_names]
 z0=origin['post_integration_poses']['tomato_sauce']['position_m'][2];trajectory=[];valid=True
 for s in samples:
  d=s['diagnostic_trace'];poses=dict(d['post_integration_poses']);poses['fl_link6']=poses['fl_link6' if active_side=='left' else 'fr_link6'];p,r=relative(poses);sp=False;fc=False;force=0.;finger_hits=[False,False]
  for c in d['solver_contacts']:
   i,j=c['geom_ids'];f=max(0.,c['force_contact_frame'][0])
   if ((i in target and j in support) or (j in target and i in support)) and f>0:sp=True
   if (i in target and j in finger) or (j in target and i in finger):fc|=f>0;force=max(force,f)
   for k,geoms in enumerate(active_geoms):
    if f>0 and ((i in target and j in geoms) or (j in target and i in geoms)):finger_hits[k]=True
  any_finger_contact=fc;fc=any(finger_hits)
  valid &= bool(s['sensor_valid']);trajectory.append((p,r,sp,fc,force,d['post_integration_poses']['tomato_sauce']['position_m'][2]-z0,any_finger_contact))
 accepted=sum(v.get('execution',{}).get('accepted',False) for v in records);complete=accepted==163 and len(samples)==4075
 streak=0;longest=0;stable=False;lift=False
 for i,(p,r,sp,fc,force,height,any_fc) in enumerate(trajectory):
  lift |= height>=criteria['lift_m'] and not sp and fc
  streak=streak+1 if fc and not sp else 0;longest=max(longest,streak)
  if streak>=501: # endpoint span >=1s at2ms, not500 endpoints spanning998ms
   tail=trajectory[i-500:i+1];a,b=tail[0][:2]
   if max(np.linalg.norm(x[0]-a) for x in tail)<=criteria['hold_translation_m'] and max(np.arccos(np.clip((np.trace(b.T@x[1])-1)/2,-1,1)) for x in tail)<=criteria['hold_rotation_rad'] and max(x[4] for x in tail)<=criteria['hold_normal_limit_N']:stable=True
 peak=max((x[4] for x in trajectory),default=0.);place=bool(trajectory and trajectory[-1][2] and not trajectory[-1][6] and abs(trajectory[-1][5])<=criteria['place_height_tolerance_m']);force_ok=peak<=criteria['peak_normal_limit_N'];qualified=complete and valid and lift and stable and place and force_ok
 latency=[v['inference_s'] for v in records if 'inference_s' in v and v.get('chunk_offset',0)==0]
 return {'complete':complete,'executed_model_ticks':accepted,'executed_sim_duration_s':accepted*.05,'sensors_valid':valid and bool(samples),'lift_30mm':bool(lift),'longest_unsupported_finger_contact_span_s':max(0,longest-1)*.002,'stable_hold_1s':bool(stable),'released_place':place,'peak_normal_N':peak,'force_bound_pass':force_ok,'qualified_task_success':bool(qualified),'inference_mean_s':float(np.mean(latency)) if latency else None,'inference_p95_s':float(np.percentile(latency,95)) if latency else None,'active_side':active_side,'contact_requirement':'any active-hand finger contact; original rule retained','scope':'Position-development v1 with20 approach ticks; unchanged task thresholds; active-hand audit, active-hand relative pose. Executed states only. Partial episodes never task success. Physics paused during inference; not realtime latency.'}

from pathlib import Path
import json,sys,numpy as np,mujoco
from cwmj_mainline_scenario import ScenarioTomatoEnv
from cwmj_slip_outcomes import relative
from validate_expert_data import validate_cwmj_capture
R=Path('/mnt/relaydata/lingbot-tactile-lab');D=R/'data/cwmj_full_trajectory_scale_v1'/sys.argv[1];audit=validate_cwmj_capture(D,D/'completion.json');identity=json.loads((D/'identity.json').read_text());e=ScenarioTomatoEnv(identity['scenario'],cameras46=True,render=False)
try:
 table=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_BODY,identity['support_body_name']);support=set(np.flatnonzero(e.model.geom_bodyid==table).tolist())
finally:e.close()
target=set(identity['target_geom_ids']);finger=set(sum(identity['finger_geom_ids'].values(),[]));samples=[]
for row in json.loads((D/'result.json').read_text())['records']:
 for s in json.loads((D/row['diagnostic_file']).read_text()):
  d=s['diagnostic_trace'];p,r=relative(d['post_integration_poses']);sp=False;fc=False;force=0.
  for c in d['solver_contacts']:
   i,j=c['geom_ids'];f=max(0.,c['force_contact_frame'][0])
   if ((i in target and j in support) or (j in target and i in support)) and f>0:sp=True
   if (i in target and j in finger) or (j in target and i in finger):fc|=f>0;force=max(force,f)
  samples.append({'stage':row['stage'],'t':d['pose_time_s'],'z':d['post_integration_poses']['tomato_sauce']['position_m'][2],'p':p,'r':r,'support':sp,'finger':fc,'force':force,'valid':s['sensor_valid']})
h=[s for s in samples if s['stage']=='hold_lift'];tail=[s for s in h if s['t']>=h[-1]['t']-1+1e-9];z0=samples[0]['z'];drift=max(float(np.linalg.norm(s['p']-tail[0]['p'])) for s in tail);angle=max(float(np.arccos(np.clip((np.trace(tail[0]['r'].T@s['r'])-1)/2,-1,1))) for s in tail)
checks={'provenance':audit['status']=='passed','all_sensor_valid':all(s['valid'] for s in samples),'lift_30mm':max(s['z'] for s in samples)-z0>=.03,'last_1s_hold_contact_no_support':len(tail)==500 and all(s['finger'] and not s['support'] for s in tail),'hold_relative_drift_5mm':drift<=.005,'hold_rotation_5deg':angle<=np.deg2rad(5),'place_back_10mm_and_supported':abs(samples[-1]['z']-z0)<=.01 and samples[-1]['support'],'released_at_end':not samples[-1]['finger']}
out={'checks':{k:bool(v) for k,v in checks.items()},'task_quality_pass':all(checks.values()),'training_approval':False,'reason':'Stage outcome gate only; action packing, masks and train manifest admission pending','metrics':{'lift_m':max(s['z'] for s in samples)-z0,'hold_drift_m':drift,'hold_rotation_rad':angle,'peak_normal_N':max(s['force'] for s in samples)},'threshold_source':'Existing run_cwmj47_tomato_task lift30mm/hold1s/drift5mm/5deg/place10mm. No calibrated hardware force claim.'};(D/'task_quality.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))

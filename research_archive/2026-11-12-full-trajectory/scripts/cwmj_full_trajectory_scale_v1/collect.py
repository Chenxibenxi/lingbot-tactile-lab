from pathlib import Path
import json,hashlib,shutil,sys
from cwmj_mainline_env import MainlineContinuousEnv
from cwmj_mainline_scenario import ScenarioTomatoEnv
from cwmj_multimodal_capture import capture_observation
from validate_expert_data import validate_cwmj_capture
import run_cwmj47_tomato_task as task
R=Path('/mnt/relaydata/lingbot-tactile-lab');P=R/'datasets/cwmj_full_trajectory_scale_v1.json';plan=json.loads(P.read_text());seed=int(sys.argv[1]);case=next(c for c in plan['cases'] if c['scenario']['seed']==seed);assert case['split']=='train','validation/test require separate explicit collection path'
O=R/'data/cwmj_full_trajectory_scale_v1'/str(seed);O.mkdir(parents=True,exist_ok=False);(O/'protocol.json').write_text(json.dumps({'reservation':case,'plan_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'warmup_ticks':1,'training_approval':False},indent=2));e=MainlineContinuousEnv(ScenarioTomatoEnv(case['scenario'],cameras46=True,render=False));e.kernel.allowed_contacts=frozenset({'asset_table','finger_object'});records=[];status='partial'
try:
 rows=task.build_command_table(task.load_46_reference())['rows'];_,warm=e.checked_step(rows[0]['actuator_target_14']);(O/'warmup.json').write_text(json.dumps(warm));assert warm['accepted'];(O/'identity.json').write_text(json.dumps(e.env.trace_identity()));e.env.render_enabled=True;pre=capture_observation(e,O,'observation_000')
 for i,row in enumerate(rows[1:]):
  if shutil.disk_usage(O).free<4*1024**3:raise RuntimeError('disk reserve reached; partial capture retained')
  obs,t=e.checked_step(row['actuator_target_14']);record={'index':i,'expert_table_tick':row['tick'],'stage':row['stage'],'pre':pre,'execution':t,'behavior_cloning_allowed':False};records.append(record)
  if not t['accepted']:break
  record['post']=capture_observation(e,O,f'observation_{i+1:03d}',observation=obs);d=O/f'diagnostic_{i:03d}.json';d.write_text(json.dumps(e.env._last_interval_substeps));record['diagnostic_file']=d.name;pre=record['post']
 if len(records)==143 and all(r['execution']['accepted'] for r in records):status='complete'
finally:
 e.close();(O/'result.json').write_text(json.dumps({'status':status,'records':records,'training_approval':False},indent=2));files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()};(O/'completion.json').write_text(json.dumps({'schema':'cwmj_capture_complete_v1','status':status,'files':files},indent=2))
if status=='complete':
 audit=validate_cwmj_capture(O,O/'completion.json');(O/'validation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
else:raise RuntimeError('incomplete physical trajectory; not admitted')

from pathlib import Path
import json,hashlib,numpy as np
R=Path('/mnt/relaydata/lingbot-tactile-lab');P=R/'datasets/cwmj_full_trajectory_scale_v1.json';D=R/'data/cwmj_full_trajectory_windows_v1';D.mkdir(exist_ok=False);plan=json.loads(P.read_text());index=[];trajectory=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for case in plan['cases']:
 if case['split']!='train':continue
 seed=case['scenario']['seed'];root=R/'data/cwmj_full_trajectory_scale_v1'/str(seed);cm=json.loads((root/'completion.json').read_text());assert cm['status']=='complete'
 for name,h in cm['files'].items():assert sha(root/name)==h,(seed,name)
 assert json.loads((root/'task_quality.json').read_text())['task_quality_pass'];rows=json.loads((root/'result.json').read_text())['records'];assert len(rows)==143
 actions=np.array([r['execution']['actual']['raw_action']['action_14d_normalized'] for r in rows],dtype=np.float32);assert actions.shape==(143,14) and np.isfinite(actions).all();file=D/f'{seed}_actions.npy';np.save(file,actions);trajectory.append({'seed':seed,'file':file.name,'sha256':sha(file),'source_completion_sha256':sha(root/'completion.json')})
 for i,row in enumerate(rows):
  assert row['index']==i and row['execution']['accepted'];meta=json.loads((root/row['pre']['metadata']).read_text());tm=meta['observation']['decision_time_s'];intervals=meta['causal_tactile_intervals'];assert 1<=len(intervals)<=3 and all(v['interval_end_s']<=tm+1e-9 for v in intervals);assert abs(tm-row['execution']['actual']['sim_time_before_s'])<1e-9
  count=min(50,len(rows)-i)
  for k in range(i,i+count):assert abs(rows[k]['execution']['actual']['sim_time_before_s']-tm-(k-i)*.05)<1e-8
  index.append({'seed':seed,'scenario_id':case['scenario_id'],'split':'train','offset':i,'valid_steps':count,'stage':row['stage'],'observation':str(root/row['pre']['metadata']),'observation_sha256':row['pre']['metadata_sha256'],'causal_intervals':len(intervals)})
(D/'index.json').write_text(json.dumps({'schema':'cwmj_full_trajectory_windows_v1','plan_sha256':sha(P),'horizon_steps':50,'action_step_s':.05,'action_semantics':'raw requested absolute targets; normalized grippers; 12 joint radians; applied governor targets remain separate observed history','padding':'zero-pad actions after valid_steps and mask every padded time/dimension; never cross trajectory','early_tactile':'left-pad missing causal intervals with zeros AND false masks; do not duplicate first interval','model_tensor_transform_validated':False,'trajectories':trajectory,'windows':index},indent=2));print(json.dumps({'trajectories':len(trajectory),'windows':len(index),'full_horizon':sum(x['valid_steps']==50 for x in index),'tail_masked':sum(x['valid_steps']<50 for x in index),'early_partial_tactile':sum(x['causal_intervals']<3 for x in index)}))

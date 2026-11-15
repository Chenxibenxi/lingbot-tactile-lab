from pathlib import Path
import json,hashlib,math,subprocess
R=Path('/mnt/relaydata/lingbot-tactile-lab');D=R/'data/cwmj_epoch138_armrole_v2'
def read(p):return json.loads(p.read_text())
rows=[]
for ep in read(D/'summary.json')['episodes']:
 case=ep['case']; row={'case':case}
 for ver in ['cwmj_fixed_dual_eval138_v2','cwmj_epoch138_chunk2_v1','cwmj_epoch138_armrole_v2']:
  d=R/'data'/ver/case;res=read(d/'result.json');m=read(d/'metrics.json');origin=read(d/'origin.json')['post_integration_poses'];inactive='fl_link6' if ep['side']=='right' else 'fr_link6';z0=origin['tomato_sauce']['position_m'][2];p0=origin[inactive]['position_m'];lift=drift=0
  for f in d.glob('diagnostic_*.json'):
   for s in read(f):
    poses=s['diagnostic_trace']['post_integration_poses'];lift=max(lift,1000*(poses['tomato_sauce']['position_m'][2]-z0));drift=max(drift,1000*math.dist(poses[inactive]['position_m'],p0))
  stop=res['records'][-1].get('execution',{}).get('preflight',{}).get('stop')
  row[ver]={'max_lift_mm':lift,'inactive_link6_max_displacement_mm':drift,'stop':stop,'metrics':m}
 d=D/case;rec=read(d/'result.json')['records'];ready=rec[0]['execution']['actual']['raw_action']['action_14d_normalized']
 for r in rec[1:]:
  a=r['arm_role'];inds=a['inactive_indices'];active=[i for i in range(14) if i not in inds]
  assert all(a['command_after_role'][i]==ready[i] for i in inds)
  assert all(a['command_after_role'][i]==a['model_command_before_role'][i] for i in active)
  actual=r['execution'].get('actual',r['execution'].get('preflight'))
  assert actual['raw_action']['action_14d_normalized']==a['command_after_role']
 c=read(d/'completion.json');f=read(d/'force_review_data.json');assert hashlib.sha256((d/'rollout.mp4').read_bytes()).hexdigest()==c['video_sha256']==f['video_sha256'];assert len(f['frames'])==c['frames'];assert all(len(x['samples'])==25 for x in f['frames'][1:])
 subprocess.run(['/home/admin001/miniconda3/bin/ffmpeg','-v','error','-i',str(d/'rollout.mp4'),'-f','null','-'],check=True)
 rows.append(row)
langs=[]
for p in (D/'session').glob('language_audit_*.json'):
 for a in read(p)['alternatives']:langs.append({'file':p.name,'tokens_differ':a['language_tokens_differ'],'max_first_action_abs_difference':max(map(abs,a['first_action_difference']))})
out={'rows':rows,'role_commands_verified':True,'videos_sha_full_decode_force_frames_verified':True,'language_audits':langs,'scope':'Four reused development cases. Role constraint is external execution logic, not learned arm selection.'};(D/'comparison_audit.json').write_text(json.dumps(out,indent=2))
for r in rows:print(r['case'],[(v,round(x['max_lift_mm'],2),round(x['inactive_link6_max_displacement_mm'],3),x['stop']) for v,x in r.items() if isinstance(x,dict)])

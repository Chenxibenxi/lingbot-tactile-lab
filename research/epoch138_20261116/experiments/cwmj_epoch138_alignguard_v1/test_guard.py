from pathlib import Path
import json,numpy as np,mujoco
from alignment_guard import AlignmentGuard
from cwmj_fixed_dual_arm_platform import FixedDualArmScenarioEnv
R=Path('/mnt/relaydata/lingbot-tactile-lab');W=Path(__file__).parent;plan=json.loads((W/'plan.json').read_text());counts=[]
for c in plan['episodes_order']:
 env=FixedDualArmScenarioEnv({'namespace':'fixed_dual_reserved_test','seed':c['seed'],'xy_offset_m':c['offset_m'],'yaw_offset_rad':0},active_side=c['side'],render=False,platform_path=R/plan['platform']);g=AlignmentGuard(env.model,c['side']);q=[float(env.data.qpos[a]) for a in g.qadr];cmd=q.copy();cmd[6]=(q[6]+.01)/.055;cmd[13]=(q[13]+.01)/.055
 out,a=g.apply(cmd,q,{},0);assert out==cmd and not a['applied']
 d=R/'data/cwmj_epoch138_gripensemble_v1'/(c['case_id']+'_r'+str(c['repeat'])+'__w3_d85');rec=json.loads((d/'result.json').read_text())['records'];success=0
 for i,r in enumerate(rec[1:11]):
  prev=rec[i]['execution']['actual'];state=prev['execution_substeps'][-1]['arm_state'];fq=json.loads((d/rec[i]['diagnostic_file']).read_text())[-1]['finger_qpos_m'];meas=[s['position_rad'] for s in state[:6]]+[fq['fl_link7']]+[s['position_rad'] for s in state[6:]]+[fq['fr_link7']];cmd=r['arm_role']['command_after_role'];out,a=g.apply(cmd,meas,{},i)
  assert out is not None,a
  assert out[6]==cmd[6] and out[13]==cmd[13]
  inactive=list(range(7)) if c['side']=='right' else list(range(7,14));assert all(out[k]==cmd[k] for k in inactive)
  if a['applied']:success+=1;assert abs(a['projected_position'][2]-(a['current_position'][2]-.001))<1e-10;assert a['ik_position_error_m']<=.0005
 # same measured and target, contact stops intervention, not physics checking
 out,a=g.apply(cmd,meas,{'sensor_valid':True,'per_finger':{g.prefix+'_link7':{'max_normal_N':.1}}},10);assert out==cmd and a['contact_latched'];env.close();counts.append({'case':c['case_id'],'projected_cases':success})
(W/'test_result.json').write_text(json.dumps(counts,indent=2));print(counts)

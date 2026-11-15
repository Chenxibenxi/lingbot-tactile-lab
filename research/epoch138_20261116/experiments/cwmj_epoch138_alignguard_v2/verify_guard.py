from pathlib import Path
import json,numpy as np,mujoco
from alignment_guard import AlignmentGuard
from cwmj_fixed_dual_arm_platform import FixedDualArmScenarioEnv
from metrics_witness import evaluate_full_task
R=Path('/mnt/relaydata/lingbot-tactile-lab');W=Path(__file__).parent;D=R/'data/cwmj_epoch138_alignguard_v2';plan=json.loads((W/'plan.json').read_text());reports=[]
for c in plan['episodes_order']:
 key=c['case_id']+'_r'+str(c['repeat'])+'__w3_d85';d=D/key;records=json.loads((d/'result.json').read_text())['records'];env=FixedDualArmScenarioEnv({'namespace':'fixed_dual_reserved_test','seed':c['seed'],'xy_offset_m':c['offset_m'],'yaw_offset_rad':0},active_side=c['side'],render=False,platform_path=R/plan['platform']);g=AlignmentGuard(env.model,c['side']);applied=[];switches=[];previous=None
 for i,r in enumerate(records[1:],1):
  prev=records[i-1]['execution']['actual'];qstate=prev['execution_substeps'][-1]['arm_state'];fq=json.loads((d/records[i-1]['diagnostic_file']).read_text())[-1]['finger_qpos_m'];meas=[x['position_rad'] for x in qstate[:6]]+[fq['fl_link7']]+[x['position_rad'] for x in qstate[6:]]+[fq['fr_link7']];audit=r['alignment_guard'];assert np.allclose(meas,audit['measured_robot_q14'],atol=1e-12);cmd,a=g.apply(r['arm_role']['command_after_role'],meas,prev['force_interval'],r['index']);assert a==audit
  if a['applied']:applied.append(r['index'])
  if previous is not None and previous['applied'] and not a['applied']:
   before=previous['output_command'];after=cmd if cmd is not None else r['arm_role']['command_after_role'];offset=0 if c['side']=='left' else 7;switches.append({'tick':r['index'],'max_target_jump_deg':float(np.max(np.abs(np.array(after[offset:offset+6])-before[offset:offset+6]))*180/np.pi)})
  previous=a
 support=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,env.table_body_name);sg=np.flatnonzero(env.model.geom_bodyid==support).tolist();m=evaluate_full_task(d,json.loads((d/'origin.json').read_text()),sg,plan['task_thresholds'],active_side=c['side']);env.close();report={'case':key,'guard_replay_exact':True,'applied_ticks':applied,'release_switches':switches,'metrics_recomputed':m};reports.append(report);print(key,'stable witnesses',len(m['stable_witness']),'switches',switches,flush=True)
(D/'guard_witness_audit.json').write_text(json.dumps(reports,indent=2))

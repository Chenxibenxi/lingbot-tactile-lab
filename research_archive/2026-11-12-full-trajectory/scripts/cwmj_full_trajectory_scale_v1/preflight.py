from pathlib import Path
import json,time,hashlib,numpy as np
from cwmj_mainline_env import MainlineContinuousEnv
from cwmj_mainline_scenario import ScenarioTomatoEnv
import run_cwmj47_tomato_task as task
R=Path('/mnt/relaydata/lingbot-tactile-lab');P=R/'datasets/cwmj_full_trajectory_scale_v1.json';plan=json.loads(P.read_text());O=R/'data/reports/cwmj_full_trajectory_preflight_v1';O.mkdir(exist_ok=False);rows=task.build_command_table(task.load_46_reference())['rows'];results=[];start=time.monotonic()
for case in plan['cases']:
 if case['scenario']['seed'] not in plan['initial_cpu_preflight_seeds']:continue
 e=MainlineContinuousEnv(ScenarioTomatoEnv(case['scenario'],cameras46=True,render=False));e.kernel.allowed_contacts=frozenset({'asset_table','finger_object'});ticks=[];failure=None
 try:
  for row in rows:
   obs,t=e.checked_step(row['actuator_target_14'])
   if not t['accepted']:failure={'tick':row['tick'],'stage':row['stage'],'execution':t};break
   sample=e.env._last_interval_substeps[-1];poses=sample['diagnostic_trace']['post_integration_poses']
   ticks.append({'tick':row['tick'],'stage':row['stage'],'time_s':float(e.env.data.time),'asset_position_m':poses['tomato_sauce']['position_m'],'contact_present':sample['contact_present'],'sensor_valid':sample['sensor_valid'],'command':row['actuator_target_14']})
  record={'seed':case['scenario']['seed'],'scenario_id':case['scenario_id'],'accepted_ticks':len(ticks),'planned_ticks':len(rows),'execution_complete':len(ticks)==len(rows),'expert_approved':False,'ticks':ticks,'failure':failure};(O/f"seed{record['seed']}.json").write_text(json.dumps(record,indent=2));results.append({k:v for k,v in record.items() if k not in ['ticks','failure']}|{'failure_stage':failure['stage'] if failure else None});(O/'progress.json').write_text(json.dumps(results,indent=2))
 finally:e.close()
result={'status':'completed','wall_s':time.monotonic()-start,'plan_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'cases':results,'scope':'CPU physical execution feasibility only. No image capture or task/expert qualification.'};(O/'result.json').write_text(json.dumps(result,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,4))
for r in results:
 t=json.loads((O/f"seed{r['seed']}.json").read_text())['ticks'];ax.plot([x['time_s'] for x in t],[x['asset_position_m'][2] for x in t],label=str(r['seed']))
ax.set(xlabel='Simulation time (s)',ylabel='Object world height (m)',title='Full expert trajectory physical preflight');ax.legend(ncol=4);fig.tight_layout();fig.savefig(O/'height.png',dpi=140);plt.close(fig)
(O/'report.md').write_text('# 完整专家轨迹批量采集预检\n\n![物体高度轨迹](height.png)\n\n'+f"8个预留训练场景，完成全部144步的数量：{sum(x['execution_complete'] for x in results)}/8。\n\n"+'只验证既定安全门槛下的物理执行，不将跑完轨迹等同于抓取成功或专家入库。原始失败预检与每步物体位姿均已保存；未采集图像，没有运行VLA或GPU训练，验证/测试划分未使用。\n\n```json\n'+json.dumps(results,indent=2)+'\n```\n');print(json.dumps(result),flush=True)

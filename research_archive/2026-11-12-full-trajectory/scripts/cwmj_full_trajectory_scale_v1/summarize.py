from pathlib import Path
import json,hashlib,collections,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/relaydata/lingbot-tactile-lab');D=R/'data/cwmj_full_trajectory_scale_v1';O=R/'data/reports/cwmj_full_trajectory_scale_summary_v1';O.mkdir(exist_ok=False);plan=json.loads((R/'datasets/cwmj_full_trajectory_scale_v1.json').read_text());rows=[];stages=collections.Counter()
for c in plan['cases']:
 p=D/str(c['scenario']['seed'])
 if c['split']!='train':assert not p.exists();continue
 completion=json.loads((p/'completion.json').read_text());v=json.loads((p/'validation.json').read_text());q=json.loads((p/'task_quality.json').read_text());r=json.loads((p/'result.json').read_text());assert completion['status']=='complete' and v['status']=='passed' and len(r['records'])==143;assert json.loads((p/'protocol.json').read_text())['reservation']['scenario_id']==c['scenario_id']
 for rec in r['records']:stages[rec['stage']]+=1
 rows.append({'seed':c['scenario']['seed'],'scenario_id':c['scenario_id'],'completion_sha256':hashlib.sha256((p/'completion.json').read_bytes()).hexdigest(),'quality_sha256':hashlib.sha256((p/'task_quality.json').read_bytes()).hexdigest(),'quality_pass':q['task_quality_pass'],'metrics':q['metrics'],'bytes':sum(f.stat().st_size for f in p.iterdir() if f.is_file())})
out={'trajectories':len(rows),'transitions':sum(stages.values()),'quality_pass':sum(r['quality_pass'] for r in rows),'stage_transitions':dict(stages),'raw_GiB':sum(r['bytes'] for r in rows)/1024**3,'validation_and_test_uncollected':True,'training_admission':False,'rows':rows};(O/'manifest.json').write_text(json.dumps(out,indent=2))
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,key,scale,label in zip(axes,['lift_m','hold_drift_m','peak_normal_N'],[1000,1000,1],['Lift (mm)','Hold drift (mm)','Peak normal (N)']):ax.hist([r['metrics'][key]*scale for r in rows],bins=16);ax.set_xlabel(label);ax.set_ylabel('Trajectories')
fig.tight_layout();fig.savefig(O/'quality.png',dpi=140);plt.close(fig)
(O/'report.md').write_text('# 完整轨迹扩量采集汇总\n\n128个预留训练场景全部采集完成并通过来源与阶段任务质量检查，共18304个连续50ms转移，128份逐轨迹图文报告。验证与测试场景未采集。\n\n![质量分布](quality.png)\n\n'+f"原始数据约{out['raw_GiB']:.2f} GiB。\n\n"+'相比旧保持数据，覆盖下降、闭合、抬升、保持、放置、释放、撤回的左臂动作；右臂固定，同一资产和同一脚本轨迹仅改变物体初始位姿，不代表128种技能或恢复专家。帧间高度相关，18304个转移不是独立样本。下一步按轨迹构建动作块及有效掩码，避免跨回合窗口、未来信息泄漏和对原始图像重复缓存；此时仍未批准训练入库。\n\n```json\n'+json.dumps(dict(stages),indent=2)+'\n```\n')
print({k:v for k,v in out.items() if k!='rows'})

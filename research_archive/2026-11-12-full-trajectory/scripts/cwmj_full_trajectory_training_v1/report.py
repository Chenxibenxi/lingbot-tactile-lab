from pathlib import Path
import json,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/relaydata/lingbot-tactile-lab');name=sys.argv[1];assert name in ['cwmj_full_trajectory_training_v1','cwmj_full_trajectory_extension_v1'];O=R/'data/reports'/name;D=R/'data'/name
status=json.loads((O/'status.json').read_text()) if (O/'status.json').exists() else {'status':'not_started'};cfg=json.loads((R/'datasets'/(name+'.json')).read_text());rows=[]
if (O/'metrics.jsonl').exists():
 for line in (O/'metrics.jsonl').read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
fig,axes=plt.subplots(1,2,figsize=(12,4));summary={}
for arm in cfg['arms']:
 xs=[x for x in rows if x.get('kind')=='train' and x['arm']==arm];summary[arm]={'updates':len(xs),'planned':cfg['steps_per_arm']}
 if xs:
  for ax,key in zip(axes,['action_loss','loss']):
   blocks=[xs[i:i+128] for i in range(0,len(xs),128)];ax.plot([b[-1]['step'] for b in blocks],[np.mean([x[key] for x in b]) for b in blocks],label=arm);ax.set(xlabel='Optimizer updates',ylabel=key)
  summary[arm]['last']=xs[-1]
for ax in axes:
 if ax.lines:ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(O/'training.png',dpi=140);plt.close(fig)
weights={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in D.glob('*.pt') if not f.is_symlink()};out={'status':status,'arms':summary,'weights':weights,'matched_final_budget':all(v['updates']==cfg['steps_per_arm'] for v in summary.values())};(O/'training_summary.json').write_text(json.dumps(out,indent=2))
(O/'report.md').write_text('# 完整轨迹触觉动作训练\n\n![训练曲线，128步均值](training.png)\n\n'+f"状态：{status.get('status')}。各组更新数量："+str({a:v['updates'] for a,v in summary.items()})+'。\n\n128条同资产脚本轨迹，18304个相关窗口；三组同初始联合触觉权重、同数据顺序和主损失噪声，视觉组屏蔽触觉、负对照替换触觉。共同触觉预训练历史限制对照解释。保留旧468辅助窗口；不声称新增恢复专家。训练损失不能替代独立闭环能力或触觉收益证据。若截止或异常导致预算不齐，不作匹配预算终态比较。\n\n完整状态、参数文件SHA和每组统计见training_summary.json。\n')
print(name,status.get('status'),{a:v['updates'] for a,v in summary.items()})

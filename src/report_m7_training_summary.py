"""All three fixed M7 repetitions against matching M6; descriptive offline only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]
DEST=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m7-training')
def report():
 records=[]
 for i in range(3):
  new=R/f'data/reports/cwmj_m7_training_r{i}';old=R/f'data/reports/cwmj_m6_training_r{i}'
  n=json.loads((new/'result.json').read_text());o=json.loads((old/'result.json').read_text());p=json.loads((new/'protocol.json').read_text());q=json.loads((old/'protocol.json').read_text())
  if n['status']!='completed_real_adapter_training_offline_pilot' or n['repeat_index']!=i:raise ValueError('complete fixed repetition required')
  if n['results']['frozen']!=o['results']['frozen'] or p['initialization']!=q['initialization'] or p['transformed_dataset_sha256']!=q['transformed_dataset_sha256']:raise ValueError('M6/M7 matching failure')
  w=n['updates']['contact_state']
  if w['steps']!=720 or sha256(Path(w['artifact']))!=w['sha256']:raise ValueError('final artifact failure')
  values={a:{s:float(v['True']) for s,v in o['summary_first10_mae'][a].items()} for a in ('zero','true','mismatch')}
  values['contact_state']={s:float(v['True']) for s,v in n['summary_first10_mae']['contact_state'].items()}
  records.append(dict(repeat=i,initialization_sha256=p['initialization']['parameter_sha256'],dataset_sha256=p['transformed_dataset_sha256'],contact_first10_mae=values,weight=w))
 if len({r['initialization_sha256'] for r in records})!=3 or len({r['dataset_sha256'] for r in records})!=1:raise ValueError('repeat contract failure')
 fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(3)
 for j,a in enumerate(('zero','true','mismatch','contact_state')):ax.bar(x+(j-1.5)*.20,[np.mean([r['contact_first10_mae'][a][str(s)] for s in range(100067,100071)]) for r in records],.19,label=a)
 ax.set_xticks(x,['repeat0','repeat1','repeat2']);ax.set(ylabel='New-validation seed mean contact MAE',title='M7 matched offline comparison | lower is better');ax.legend();DEST.mkdir(parents=True,exist_ok=True);fig.savefig(DEST/'all_repeats.png',dpi=160);plt.close(fig)
 summary=dict(status='three_repetitions_completed',new_optimizer_steps=2160,matched_initialization_data_frozen_validation=True,closed_loop_benefit_proven=False,records=records)
 for path in (DEST/'summary.json',R/'data/reports/cwmj_m7_training_summary.json'):path.write_text(json.dumps(summary,indent=2))
 rows=[]
 for r in records:
  for s in sorted(r['contact_first10_mae']['contact_state']):rows.append('| '+str(r['repeat'])+' | '+s+' | '+' | '.join(f"{r['contact_first10_mae'][a][s]:.6f}" for a in ('zero','true','mismatch','contact_state'))+' |')
 text='''# M7 仅接触状态训练：三次重复图文汇总

三次固定初始化、每次720步，共2160个新优化步；旧M6三个条件不重训、不覆盖。逐重复初始参数、转换数据指纹和冻结模型验证结果与M6精确一致，三次初始化各异。没有验证选权重、最好重复筛选或离线通过门。

![匹配离线比较](all_repeats.png)

| 重复 | 验证seed | zero | true | mismatch | contact_state |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## 解释与局限

图中只对新验证67–70按seed等权平均；旧验证49单列于表，不混入新验证平均。指标是实际前10动作的归一化MAE，不是任务成功率；重叠窗口不是独立环境样本。仅接触bit除去幅值、侧别和窗口内顺序，但保留coverage/age/valid。本批没有添加恢复监督或改变反馈频率。正负变化均须在固定新seed87–102的闭环对照检验，不能据此宣称因果触觉收益。

## 方法、复现与下一步

冻结154800基座，FP32适配器与M6同容量；训练源402窗口，固定验证40窗口。AdamW1e-4、weight_decay.01、clip1，教师补全未知H50尾部且不计loss。协议docs/CWMJ_M7_TRAINING_PROTOCOL_ZH.md，来源/代码pin与权重SHA在各protocol.json及本summary.json。运行work/run_cwmj_m7_training.sh 0/1/2逐轮串行。

'''+ '\n'.join(f'- [重复{i}完整训练报告](repeat_{i}/arm_contact_state/report.md)' for i in range(3))+'\n'
 (DEST/'report.md').write_text(text);print('M7_TRAINING_SUMMARY_COMPLETE')
if __name__=='__main__':report()

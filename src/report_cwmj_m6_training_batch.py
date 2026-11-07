"""Per-repeat and all-repeat illustrated reports; never select a winning checkpoint."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];DEST=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m6-training')
def report_repeat(repeat):
 out=R/f'data/reports/cwmj_m6_training_r{repeat}';r=json.loads((out/'result.json').read_text());p=json.loads((out/'protocol.json').read_text());seeds=sorted(r['summary_first10_mae']['true']);arms=['zero','true','mismatch'];fig,ax=plt.subplots(figsize=(10,4),layout='constrained');x=np.arange(len(seeds))
 for j,arm in enumerate(arms):ax.bar(x+(j-1)*.25,[r['summary_first10_mae'][arm][s]['True'] for s in seeds],.24,label=arm)
 ax.set_xticks(x,seeds);ax.set(ylabel='Contact first10 normalized MAE',title=f'M6 repetition {repeat} | fixed 720-step checkpoints');ax.legend();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 rows=[]
 for seed in seeds:
  v={a:r['summary_first10_mae'][a][seed]['True'] for a in ['zero','true','mismatch','true_input_zero','true_input_mismatch']}
  rows.append(f"| {seed} | "+' | '.join(f'{v[a]:.6f}' for a in v)+' |')
 text=f'''# M6训练重复{repeat}：完整图文报告

## 方法与结果

402个训练窗口来自7个训练seed；历史验证49单列，4个新验证67–70各8个预定窗口。三臂各720步，共2160优化步；无验证选权重、无早停。初始化seed={p['initialization_seed']}，参数SHA=`{p['initialization']['parameter_sha256']}`；输出投影初始严格零，基座154800冻结，FP32适配器。

新验证全部seed的预定离线支持门通过：**{r['predeclared_offline_support']}**。该门要求true接触MAE比zero和mismatch各改善至少5%，且同权重置零/错配均变差。它是离线信息线索，不能证明闭环收益；旧49不作为新验证门。

![三组对照](comparison.png)

| 验证seed | zero | true | mismatch | true权重输入zero | true权重输入mismatch |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

## 解释边界与下一步

窗口不是独立样本；三次训练随机重复全部报告，不挑最好的一次。归一化与错配供体只来自训练集；未知H50尾标签不参与loss，教师补全仅使用当前观测。单瓶替换场景不能外推原20瓶分布。按协议无论离线门是否通过，都完成固定独立确认seed71–86和运行时输入消融；反馈频率调整另立实验。

## 逐实验图文与复现

'''+ '\n'.join(f'- [{a} 训练](arm_{a}/report.md)' for a in arms)+f'''

协议 `{R}/docs/CWMJ_M6_TRAINING_PROTOCOL_ZH.md`；训练源、供体、参数、权重SHA在protocol.json、donor_mapping.json、result.json。每臂完整损失和显存统计见子报告。运行脚本work/run_cwmj_m6_training.sh {repeat}；原源码/checkpoint只读，自有GPU租约退出释放，不重启Strata。
'''
 (out/'report.md').write_text(text);dest=DEST/f'repeat_{repeat}';dest.mkdir(parents=True,exist_ok=True)
 for name in ['comparison.png','report.md','result.json','protocol.json','donor_mapping.json','training_log.json']:shutil.copy2(out/name,dest/name)
 print('M6_REPEAT_REPORTED',repeat,r['predeclared_offline_support'],flush=True)
def report_all():
 rs=[json.loads((R/f'data/reports/cwmj_m6_training_r{i}/result.json').read_text()) for i in range(3)];ps=[json.loads((R/f'data/reports/cwmj_m6_training_r{i}/protocol.json').read_text()) for i in range(3)]
 if len({p['initialization']['parameter_sha256'] for p in ps})!=3:raise ValueError('training initializations not distinct')
 if len({p['training_manifest_sha256'] for p in ps})!=1:raise ValueError('different data manifest across repetitions')
 if len({p['transformed_dataset_sha256'] for p in ps})!=1:raise ValueError('transformed inputs/targets changed across repetitions')
 if any(r['results']['frozen']!=rs[0]['results']['frozen'] for r in rs[1:]):raise ValueError('frozen evaluation changed across independent loads')
 summary={'repeats':3,'all_offline_gates_passed':all(r['predeclared_offline_support'] for r in rs),'offline_gates':[r['predeclared_offline_support'] for r in rs],'distinct_initializations':True,'frozen_validation_exact_equal':True,'closed_loop_improvement_proven':False}
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(3)
 for j,a in enumerate(['zero','true','mismatch']):ax.bar(x+(j-1)*.25,[np.mean([r['summary_first10_mae'][a][str(s)]['True'] for s in r['new_validation_seeds']]) for r in rs],.24,label=a)
 ax.set_xticks(x,['repeat0','repeat1','repeat2']);ax.set(ylabel='Mean over 4 new validation seeds',title='M6 all fixed repetitions | contact first10 MAE');ax.legend();DEST.mkdir(exist_ok=True);fig.savefig(DEST/'all_repeats.png',dpi=160);plt.close(fig)
 (DEST/'summary.json').write_text(json.dumps(summary,indent=2));(R/'data/reports/cwmj_m6_training_summary.json').write_text(json.dumps(summary,indent=2))
 text=f'''# M6三次训练重复：完整汇总

三次重复、每次三臂720步，共6480优化步完成。三个初始化参数SHA各异，跨加载冻结验证输出指标逐项完全相同。每次预定离线门：{summary['offline_gates']}；全部通过={summary['all_offline_gates_passed']}。全部模型保留，不挑最佳重复或checkpoint。

![四个新验证seed平均指标](all_repeats.png)

图对4个新验证seed等权平均，仅描述离线动作误差，不能视作闭环成功率。旧验证49的结果保留在各重复报告，未参与新验证门。当前仍未证明稳定任务收益，按协议继续独立seed71–86及同权重输入消融；原比赛分布、真实硬件和失败恢复均不能由此外推。

'''+ '\n'.join(f'- [重复{i}方法、结果、三臂图文与复现](repeat_{i}/report.md)' for i in range(3))+'\n'
 (DEST/'report.md').write_text(text);print('M6_ALL_REPEATS_REPORTED',json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,choices=(0,1,2));p.add_argument('--all',action='store_true');a=p.parse_args()
 if a.all:report_all()
 elif a.repeat is not None:report_repeat(a.repeat)
 else:p.error('choose repeat or all')

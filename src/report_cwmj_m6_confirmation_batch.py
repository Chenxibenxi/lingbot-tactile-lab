"""Complete matrix report; require every planned attempt and paired source checks."""
from pathlib import Path
import json,shutil
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from audit_cwmj_m6_pair import audit_pair,CONDITIONS
from paired_seed_statistics import summarize_paired_success
R=Path(__file__).resolve().parents[1];DEST=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m6-confirmation')
def report():
 root=R/'data/reports/cwmj_m6_confirmation';rows=[];eligible=[];outcomes=[];exclusions=[];matrix=np.full((12,16),-1,dtype=int)
 for j,seed in enumerate(range(100071,100087)):
  seed_results=[];audits=[];refs=[]
  for repeat in range(3):
   trials={a:R/f'data/eval/cwmj_m6_confirm_r{repeat}_{seed}_{a}' for a in CONDITIONS};audit=audit_pair(trials,seed,repeat);audits.append(audit);seed_results.append([])
   if audit['checks'] and not all(audit['checks'].values()):raise ValueError(f'pair gate failed {seed}/{repeat}')
   for i,a in enumerate(CONDITIONS):
    r=json.loads((trials[a]/'result.json').read_text());full=audit['full_episodes'][a]
    rows.append({'seed':seed,'repeat':repeat,'condition':a,'full':full,'success':r.get('success'),'actions':r['actions'],'termination':r.get('termination',r.get('exception'))})
    matrix[repeat*4+i,j]=int(r['success']) if full else -1;seed_results[-1].append(int(bool(r.get('success'))))
   if audit.get('eligible_complete_pair'):refs.append(trials['zero']/'rollout.hdf5')
  cross=False
  if len(refs)==3:
   with h5py.File(refs[0]) as first:
    base={k:first[k][0] for k in ['images/head','images/left','images/right','state_14']};commands=first['applied/command_14'][:10]
   cross=True
   for path in refs[1:]:
    with h5py.File(path) as h:cross=cross and all(np.array_equal(h[k][0],v) for k,v in base.items()) and np.array_equal(h['applied/command_14'][:10],commands)
   if not cross:raise ValueError(f'cross-repetition initial gate failed {seed}')
  if cross:eligible.append(seed);outcomes.append(seed_results)
  else:exclusions.append({'seed':seed,'reason':'not all repetition/condition episodes complete and technically comparable','audits':audits})
 stats=summarize_paired_success(np.asarray(outcomes,dtype=int).reshape(len(eligible),3,4),eligible);stats.update(all_attempts=rows,exclusions=exclusions,planned_episodes=192)
 root.mkdir(exist_ok=True);(root/'summary.json').write_text(json.dumps(stats,indent=2));DEST.mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(14,7),layout='constrained');ax.imshow(matrix,cmap=ListedColormap(['#c7cad0','#dda984','#6fae91']),vmin=-1,vmax=1,aspect='auto');ax.set_xticks(range(16),range(100071,100087),rotation=45,ha='right');ax.set_yticks(range(12),[f'r{r} / {a}' for r in range(3) for a in CONDITIONS]);ax.set_title('All 192 fixed attempts | green success, orange full failure, gray excluded')
 for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,'—' if v<0 else str(v),ha='center',va='center',fontsize=8)
 fig.savefig(root/'matrix.png',dpi=160);plt.close(fig)
 def fmt(x):return 'NA' if x is None else f'{x:.4f}'
 table='\n'.join(f"| {c['comparison']} | {fmt(c['success_rate_difference'])} | {c['seed_cluster_bootstrap_95']} | {c['exact_two_sided_p']:.5f} | {c['holm_adjusted_p']:.5f} |" for c in stats['contrasts'])
 trials='\n'.join(f"| {r['seed']} | {r['repeat']} | [{r['condition']}](r{r['repeat']}_{r['seed']}_{r['condition']}/report.md) | {r['success']} | {r['actions']} | {r['termination']} |" for r in rows)
 text=f'''# M6固定独立闭环确认：完整图文汇报

## 结果与预定判据

192个计划回合均有终态记录；完整且通过全部技术配对、跨重复起始一致性门的独立seed为{len(eligible)}/16。预先规定的稳定收益判据满足：**{stats['predeclared_stable_benefit_criterion_met']}**。独立单位是seed，三次训练重复先在每个seed内平均；不能当作48个独立环境样本。缺失/截断/初始化异常未冒充完整失败。

![完整实验矩阵](matrix.png)

| true相对条件 | 成功率差 | seed簇bootstrap95%区间 | 双侧sign-flip p | Holm p |
|---|---:|---|---:|---:|
{table}

## 方法与解释边界

16个预定新seed71–86、三次训练重复、四条件；固定最后720步权重，未挑最好重复。原400动作、每10动作反馈、10步去噪、相同seed/request噪声，CWMJ v2资产与原任务判据不变。真实触觉仅过去窗口，错配仅冻结训练供体，同权重置零保留mask/age/valid。每组初始三相机/状态/首10命令/首窗invalid/资产核验；同权重在首次接触前全部命令一致。

稳定收益门还要求至少12完整seed、对zero训练至少+10个百分点、其余两个差值>0、三个区间下界>0、三个Holm p<.05，并且每个训练重复true-zero差非负。不满足时表示未证明预定强度的稳定收益，不等于证明两方法等效或所有触觉方法无效。sign-flip依赖配对seed差的对称性；bootstrap条件于三组已训练模型，不能穷尽训练随机性。

所有排除详见summary.json：{[e['seed'] for e in exclusions]}。原20瓶分布、真实硬件、力学安全与失败恢复不由单瓶替换场景外推。成功回合动作数只作次要描述，400步失败不是完成用时。

## 所有逐回合图文报告

| seed | 重复 | 条件/报告 | 成功 | 动作 | 终止 |
|---|---:|---|---|---:|---|
{trials}

## 复现与下一步

协议 `{R}/docs/CWMJ_M6_TRAINING_PROTOCOL_ZH.md`；冻结checkpoint/代码/供体在data/reports/cwmj_m6/confirmation_manifest.json，原始回合在data/eval/cwmj_m6_confirm_*。训练、测试、运行时输入消融分别解释；根据完整证据决定是否需要另立反馈频率、历史窗口或恢复监督实验，不改写本批模型与结果。原仓库/checkpoint只读，自有GPU租约，未自动重启Strata。
'''
 (root/'report.md').write_text(text)
 for name in ['summary.json','matrix.png','report.md']:shutil.copy2(root/name,DEST/name)
 print('M6_CONFIRMATION_SUMMARY',len(eligible),stats['predeclared_stable_benefit_criterion_met'],flush=True)
if __name__=='__main__':report()

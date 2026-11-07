"""Complete M4 comparison report after all fixed arms, no posthoc selection."""
import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];ROOT=R/'data/reports/cwmj_m4_training_v1'
def report():
    r=json.loads((ROOT/'result.json').read_text());s=r['summary_first10_mae'];seeds=sorted(next(iter(s.values())).keys())
    arms=['frozen','zero','true','mean','mismatch','true_input_zero','true_input_mean','true_input_mismatch']
    fig,axs=plt.subplots(1,len(seeds),figsize=(max(9,6*len(seeds)),4),squeeze=False,layout='constrained')
    for ax,seed in zip(axs.flat,seeds):
        values=[s[a][seed]['True'] for a in arms];ax.bar(range(len(arms)),values,color=['#999999','#719ece','#328c62','#dfa849','#c57a73','#afc5ba','#d4bd8f','#cba7a3']);ax.set_xticks(range(len(arms)),arms,rotation=30,ha='right',fontsize=8)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.4f}',ha='center',va='bottom',fontsize=8)
        ax.set_ylim(0,max(values)*1.2);ax.set_ylabel('Contact first10 normalized MAE');ax.set_title(f'CWMJ validation seed {seed} (not original bottle distribution)')
    fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    rows=[]
    for a in arms:
        for seed in seeds:rows.append(f"| {a} | {seed} | {s[a][seed]['True']:.6f} | {s[a][seed]['False']:.6f} |")
    changes={seed:{a:100*(1-s['true'][seed]['True']/s[a][seed]['True']) for a in ['zero','mean','mismatch']} for seed in seeds}
    (ROOT/'relative_changes.json').write_text(json.dumps(changes,indent=2))
    text=f'''# M4 CWMJ四组正式适配器训练：完整图文报告

## 主要结果

四组各完成360个真实AdamW优化步，共1440步。新场景只含CWMJ单酒瓶双侧；独立训练seed数={r['independent_train_seeds']}，验证seed数={r['independent_validation_seeds']}，只含一个优化随机初始化。预登记探索性触觉信息支持={r['predeclared_offline_support']}；均值干预子条件满足={r['temporal_support']}。主效用条件未通过时，该子条件不能解释为触觉时序收益；联合条件为{r['predeclared_offline_support'] and r['temporal_support']}。**这些离线条件不等于任务成功率提升，后续独立闭环必须单独报告。**

![完整离线对照与干预](comparison.png)

| 条件 | 验证seed | 接触前10动作MAE | 非接触前10动作MAE |
|---|---|---:|---:|
'''+ '\n'.join(rows)+f'''

真实触觉相对各对照的误差改善百分比（负值表示更差）：

```json
{json.dumps(changes,indent=2)}
```

## 方法和数据

固定train100045/100046，validation100049；两个训练回合123个因果窗口、44个过去接触窗口。验证按接触分层固定4+4窗口。未读取原场景或闭环seed作训练。原始数据均通过同回合物理时钟/实际take_action/HDF5/完成SHA，额外绑定CWMJ scenario.json。失败47/50及预检拒绝48不补换。

四组相同初始化、按seed和接触层平衡的360次抽样、相同noise/time；AdamW1e-4、weight_decay.01、clip1、batch1。154800基座BF16冻结、adapter65,664参数FP32。归一化只拟合train；teacher仅依据当前观测产生末尾H50插补，未知位置不计loss，推理不传终止长度。固定最后一步，无验证集选checkpoint或调参。

zero保留mask/age、历史特征置零；true为真实过去触觉；mean为每手指按覆盖加权的原始历史均值；mismatch由另一个训练seed同接触层提供确定性donor。错配不读取动作标签，验证永不提供donor。true还做置零/均值/错配输入干预；donor映射完整保存。

## 解释边界

时间均值仅保留原始代理量均值，经过log归一化后统计量未必相同，因此其优劣不是纯时序信息的独立因果证明。跨seed错配保持接触层，但其他幅度/位置属性变化；需要联合对照与干预解释。窗口重叠不能扩充独立seed数。一个验证seed可能受偶然性和专家轨迹差异影响，不支持总体泛化断言。

新资产的几何、视觉、质量和惯量同时改变，不能将该场景得分并入原20瓶比赛成功率。训练loss与离线动作误差不能代替任务成功。下一步不改权重/步数，执行预登记51–54、zero/true各一次，失败与初始化/资源异常分开报告。

## 逐组完整报告

- [zero](arm_zero/report.md)
- [true](arm_true/report.md)
- [mean](arm_mean/report.md)
- [mismatch](arm_mismatch/report.md)

## 复现证据

protocol.json/训练manifest绑定源数据、配置、归一化及代码SHA；result.json记录每个checkpoint路径/SHA、步数和显存；training_log.json记录每步loss/grad和实际sample；donor_mapping.json绑定错配来源。训练中验证基座无梯度、有限loss/grad、22GiB显存上限、无效哨兵不影响forward。模型加载器此前已通过独立进程逐元素输出一致性门；本轮272项回归271通过、1跳过。

完整私有目录：`{ROOT}`。每个GPU任务释放自有租约，不自动启动Strata。原比赛仓库、checkpoint保持只读。
'''
    (ROOT/'report.md').write_text(text);dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m4-training');dest.mkdir(exist_ok=True)
    for p in ROOT.iterdir():
        if p.is_file():shutil.copy2(p,dest/p.name)
    print(json.dumps({'information_support':r['predeclared_offline_support'],'temporal_support':r['temporal_support'],'changes':changes}))
if __name__=='__main__':report()

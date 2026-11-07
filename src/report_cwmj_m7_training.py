"""Save an illustrated immutable arm report immediately after fixed-step eval."""
import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def report_arm(out,arm,results,updates,logs,protocol):
    root=out/f'arm_{arm}';root.mkdir(exist_ok=False)
    seeds=sorted({r['seed'] for r in results[arm]});summary={}
    for name in ['frozen',arm]:
        summary[name]={str(s):{str(c):float(np.mean([r['mae_first10'] for r in results[name] if r['seed']==s and r['contact']==c])) for c in [False,True]} for s in seeds}
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    axes[0].plot([r['step'] for r in logs],[r['loss'] for r in logs],alpha=.5,lw=.7)
    if len(logs)>=20:axes[0].plot(np.arange(20,len(logs)+1),np.convolve([r['loss'] for r in logs],np.ones(20)/20,mode='valid'),label='20-step mean');axes[0].legend()
    axes[0].set(xlabel='Optimizer step',ylabel='Masked flow loss',title=f'{arm}: fixed 720-step training')
    x=np.arange(len(seeds));axes[1].bar(x-.18,[summary['frozen'][str(s)]['True'] for s in seeds],.35,label='Frozen vision');axes[1].bar(x+.18,[summary[arm][str(s)]['True'] for s in seeds],.35,label=arm)
    axes[1].set_xticks(x,[str(s) for s in seeds]);axes[1].set(ylabel='Contact first10 normalized MAE',title='Held-out seed inference (lower is better)');axes[1].legend()
    fig.savefig(root/'training_and_validation.png',dpi=160);plt.close(fig)
    (root/'result.json').write_text(json.dumps({'arm':arm,'training':updates[arm],'summary':summary,'protocol':protocol},indent=2))
    table='\n'.join(f"| {s} | {summary['frozen'][str(s)]['True']:.6f} | {summary[arm][str(s)]['True']:.6f} | {summary[arm][str(s)]['False']:.6f} |" for s in seeds)
    text=f'''# M7 接触状态训练实验：重复{protocol['repeat_index']} / {arm}

## 方法与目的

独立CWMJ单酒瓶场景；比较触觉信息是否有用。此报告是一个固定720步arm的结果，不做验证集选checkpoint。本组与既有M6对应重复匹配数据、初始化、抽样与噪声，基座154800 BF16冻结，adapter FP32 65,664参数；AdamW1e-4/weight_decay0.01，batch1、clip1，按seed与接触层平衡抽样。归一化仅训练回合。末尾H50由共同冻结教师补全且排除未知标签loss，终止长度不进入推理。

## 结果

完成优化步：{updates[arm]['steps']}；改变参数张量：{updates[arm]['changed_tensors']}；峰值分配显存：{updates[arm]['peak_allocated_bytes']/1024**3:.2f} GiB。权重SHA256：`{updates[arm]['sha256']}`。

![训练与独立验证](training_and_validation.png)

| 验证seed（49为历史验证，其余新验证） | 冻结视觉接触MAE | 本arm接触MAE | 本arm非接触MAE |
|---|---:|---:|---:|
{table}

## 解释与局限

采用纯sample_actions、固定10步去噪，报告前最多10条真实动作MAE。训练loss不能代替闭环成功。窗口相互重叠，独立样本单位为seed；本报告只覆盖一次训练随机重复，全部三次完成后才汇总稳定性。单arm相对冻结视觉变化可能来自适配器容量与微调，不能归因为触觉，须与既有M6对应重复以及新seed闭环对照比较。本批不检验触觉时序顺序优势。本组仅保留一个窗口级接触bit，训练与推理共用转换，不保留力幅值、侧别或窗口内时序。

新场景外观/几何/质量/惯量与原20瓶场景不同，不能合并成功率。资产已通过落稳和本批专家验证，但不是物理硬件传感器验证。下一步完成全部三次重复，再按新协议在未消费seed上比较；不复用M6测试seed确认新收益。

## 复现

本arm权重：`{updates[arm]['artifact']}`；同目录optimizer状态与metadata保留。完整训练曲线、源SHA与归一化见批目录 `{out}` 的training_log.json/protocol.json；donor映射在批结束时保存。基座无梯度、有限loss/grad、显存和补全哨兵不变检查在运行时执行。此前加载器的跨进程一致性门仍为先决条件。
'''
    (root/'report.md').write_text(text)
    dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m7-training')/f"repeat_{protocol['repeat_index']}"/f'arm_{arm}';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(root,dest)

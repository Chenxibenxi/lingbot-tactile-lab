"""One illustrated report per completed duration checkpoint; no model selection."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report_node(node, publish=True):
    node=Path(node)
    result=json.loads((node/'result.json').read_text())
    logs=json.loads((node/'training_log.json').read_text())
    seeds=sorted({x['seed'] for x in result['validation']})
    def mean(rows,s,c):
        return float(np.mean([x['mae_first10'] for x in rows if x['seed']==s and x['contact']==c]))
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    axes[0].plot([x['step'] for x in logs],[x['loss'] for x in logs],alpha=.3,lw=.5)
    if len(logs)>=40:
        axes[0].plot(np.arange(40,len(logs)+1),np.convolve([x['loss'] for x in logs],np.ones(40)/40,mode='valid'))
    axes[0].set(xlabel='Optimizer step',ylabel='Masked flow loss',title='Training (40-step moving mean)')
    seen=set();coverage=[]
    for row in logs:
        seen.add((row['seed'],row['frame']));coverage.append(len(seen))
    axes[1].plot(range(1,len(logs)+1),coverage)
    axes[1].axhline(result['train_windows'],color='grey',ls='--')
    axes[1].set(xlabel='Optimizer step',ylabel='Distinct training windows',title='Sampling coverage')
    x=np.arange(len(seeds))
    axes[2].bar(x-.2,[mean(result['frozen'],s,True) for s in seeds],.4,label='Frozen')
    axes[2].bar(x+.2,[mean(result['validation'],s,True) for s in seeds],.4,label=result['arm'])
    axes[2].set_xticks(x,[str(s) for s in seeds],rotation=35)
    axes[2].set(ylabel='First10 contact MAE',title='Fixed validation seeds');axes[2].legend()
    fig.savefig(node/'training_validation.png',dpi=150);plt.close(fig)
    rows='\n'.join(f"| {s} | {mean(result['frozen'],s,True):.6f} | {mean(result['validation'],s,True):.6f} | {mean(result['validation'],s,False):.6f} |" for s in seeds)
    text=f'''# M8 训练时长实验：r{result['repeat']} / {result['arm']} / {result['steps']} 步

## 方法与数据

保持 M6 的七个训练 seed、402 个窗口及五个验证 seed 不变。冻结 154800 主模型，仅训练 FP32 触觉适配器；AdamW 学习率 1e-4，权重衰减 0.01，梯度裁剪 1。按 seed 和接触层有放回采样，四条件内初始化、采样和流匹配噪声配对。固定保存 720/1440/2880 步，不挑选最好节点。未知终端标签用冻结教师填充且不计入损失。

## 结果

完成 {result['steps']} 步，覆盖 {result['unique_windows']}/{result['train_windows']} 个窗口；峰值显存 {result['peak_bytes']/1024**3:.2f} GiB。720 步历史权重和验证输出逐项一致：{result['historical_720_exact']}。该校验是继续增加步数的前置条件。

![训练、采样覆盖与固定验证](training_validation.png)

| 验证 seed | 冻结模型接触 MAE | 当前模型接触 MAE | 当前模型非接触 MAE |
|---|---:|---:|---:|
{rows}

## 失败分析与局限

本节点完成表示计算和复现检查通过，不代表任务成功率改善。训练损失随采样和噪声变化；窗口相关，独立统计单位是 seed。图中误差是归一化动作误差，不能解释为物理力，也不能替代闭环成功率。增加步数只重复现有分布，不补充恢复示范；完整触觉必须与接触状态和无触觉对照比较后才能归因。全条件、全重复完成前不作稳定性结论。

## 复现与下一步

权重：`{result['artifact']}`，SHA256 `{result['sha256']}`。本目录 result.json、training_log.json 保存数值；上级 protocol.json 保存数据源和冻结清单摘要。验证前后恢复随机状态和训练模式，节点不覆盖旧权重。下一步在预先固定的新 seed 上运行配对闭环，保存每回合图文，并由本地 Strata 提供独立辅助分析；最终研究结论仍需人工复核。
'''
    (node/'report.md').write_text(text)
    if publish:
        dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m8-training')/node.parent.name/node.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(node,dest)

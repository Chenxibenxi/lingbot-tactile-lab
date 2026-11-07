"""Offline descriptive duration curve; independent of frozen training/eval code."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_expert_data import sha256

R=Path(__file__).resolve().parents[1]


def report(repeat,arm):
    if type(repeat) is not int or repeat not in (0,1,2) or arm not in ('zero','true','mismatch','contact_state'):
        raise ValueError('fixed training condition required')
    source=R/f'data/reports/cwmj_m8_training_r{repeat}_{arm}'
    result=json.loads((source/'result.json').read_text())
    nodes=result['nodes'];steps=[720,1440,2880]
    if result['status']!='completed' or [n['steps'] for n in nodes]!=steps:raise ValueError('all duration nodes must finish')
    identities=None;summary=[]
    for node in nodes:
        if node['repeat']!=repeat or node['arm']!=arm or not node['historical_720_exact']:
            raise ValueError('training identity/historical gate mismatch')
        if sha256(Path(node['artifact']))!=node['sha256']:raise ValueError('trained weight changed')
        validation=node['validation']
        current=[(r['seed'],r['frame'],r['contact']) for r in validation]
        if identities is not None and current!=identities:raise ValueError('validation selection changed')
        identities=current;seeds=sorted({r['seed'] for r in validation})
        values={str(contact):[float(np.mean([r['mae_first10'] for r in validation if r['seed']==s and r['contact']==contact])) for s in seeds] for contact in (False,True)}
        if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('missing/nonfinite validation stratum')
        summary.append(dict(steps=node['steps'],seeds=seeds,seed_mae=values,
                            contact_mean=float(np.mean(values['True'])),noncontact_mean=float(np.mean(values['False'])),
                            unique_windows=node['unique_windows'],train_windows=node['train_windows']))
    out=R/f'data/reports/cwmj_m8_duration_r{repeat}_{arm}';out.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,contact,title in zip(axes[:2],(True,False),('Contact validation','No-contact validation')):
        for i,s in enumerate(summary[0]['seeds']):
            ax.plot(steps,[x['seed_mae'][str(contact)][i] for x in summary],alpha=.4,marker='.',label=str(s))
        ax.plot(steps,[np.mean(x['seed_mae'][str(contact)]) for x in summary],color='black',lw=2,marker='o',label='Seed mean')
        ax.set(xlabel='Optimizer steps',ylabel='First10 normalized action MAE',title=title);ax.set_xticks(steps);ax.legend(fontsize=6)
    axes[2].plot(steps,[x['unique_windows']/x['train_windows'] for x in summary],marker='o')
    axes[2].set(xlabel='Optimizer steps',ylabel='Fraction of training windows seen',ylim=(0,1.05),title='Sampling coverage');axes[2].set_xticks(steps)
    fig.savefig(out/'duration_curve.png',dpi=150);plt.close(fig)
    evidence=dict(source_result=str(source/'result.json'),source_sha256=sha256(source/'result.json'),repeat=repeat,arm=arm,nodes=summary,closed_loop_benefit_proven=False)
    (out/'summary.json').write_text(json.dumps(evidence,indent=2))
    rows='\n'.join(f"| {x['steps']} | {x['contact_mean']:.6f} | {x['noncontact_mean']:.6f} | {x['unique_windows']}/{x['train_windows']} |" for x in summary)
    (out/'report.md').write_text(f'''# M8 时长学习曲线：r{repeat} / {arm}

同一初始化、固定训练集及连续采样序列；720、1440、2880三个预定节点均已完成。每个验证seed先在接触/非接触层内平均，再按seed等权平均。细线显示各seed，黑线为平均；这些线不是独立训练重复。

![固定数据训练时长曲线](duration_curve.png)

| 步数 | 接触验证MAE | 非接触验证MAE | 已覆盖训练窗口 |
|---|---:|---:|---:|
{rows}

本图是离线动作误差和采样覆盖的描述，不能替代闭环成功率。单条件、单训练重复无法将变化归因于触觉；不选择误差最低的checkpoint。即使覆盖全部窗口，也不代表场景充分或已经收敛。失败分析须结合全部条件、三次重复和预注册的新seed闭环；恢复数据扩充另作实验。

源结果：`{source/'result.json'}`，SHA256 `{evidence['source_sha256']}`。各节点完整方法、训练日志、权重和逐seed验证见相应step目录。原始权重只读，当前图未改变训练或评测协议。下一步继续完整矩阵，以预定720/2880节点进行闭环比较。
''')
    dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m8-training')/f'duration_r{repeat}_{arm}'
    shutil.copytree(out,dest)
    print(dest/'report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,required=True);p.add_argument('--arm',required=True);a=p.parse_args();report(a.repeat,a.arm)

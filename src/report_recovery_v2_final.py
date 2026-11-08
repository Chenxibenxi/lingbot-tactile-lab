"""Post-run fixed-denominator report; refuses a live/incomplete queue."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_recovery_development_v2_final import audit_seed
from validate_expert_data import sha256


def report(root):
    root=Path(root)
    queue=root/'data/reports/recovery_development_v2/queue_result.json'
    q=json.loads(queue.read_text())
    if q.get('status')!='completed':raise ValueError('queue not terminal-complete')
    expected={(s,k) for s in range(100123,100127) for k in ('prefix_generation','expert_control','left_open_once_recovery')}
    actual=[(a['seed'],a['kind']) for a in q['attempts']]
    if len(actual)!=12 or set(actual)!=expected:raise ValueError('incomplete fixed attempt matrix')
    audits=[audit_seed(root,seed) for seed in range(100123,100127)]
    control=sum(a['outcomes']['expert_control'] is True for a in audits)
    recovery=sum(a['outcomes']['left_open_once_recovery'] is True for a in audits)
    technical=sum(a['technical_complete_pair'] for a in audits)
    gate=technical==4 and control>=3 and recovery>=2
    out=root/'data/reports/recovery_development_v2_final';out.mkdir(exist_ok=False)
    summary=dict(audits=audits,control_successes=control,recovery_successes=recovery,
                 technical_complete_pairs=technical,fixed_seed_denominator=4,development_gate=gate,
                 training_approval=False,queue_sha256=sha256(queue),analysis_sha256=sha256(Path(__file__)),
                 audit_code_sha256=sha256(Path(__file__).with_name('audit_recovery_development_v2_final.py')))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    matrix=np.array([[np.nan if a['outcomes'][k] is None else float(a['outcomes'][k])
                      for k in ('expert_control','left_open_once_recovery')]+[float(a['technical_complete_pair'])] for a in audits])
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('lightgrey')
    ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set_xticks(range(3),['Control success','Recovery success','Technical pair passed'])
    ax.set_yticks(range(4),[str(a['seed']) for a in audits])
    for i in range(4):
        for j in range(3): ax.text(j,i,'Unknown' if np.isnan(matrix[i,j]) else ('Yes' if matrix[i,j] else 'No'),ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    ax.set_title('Fixed recovery development v2: all four seeds')
    fig.savefig(out/'matrix.png',dpi=150);plt.close(fig)
    table='\n'.join(f"| {a['seed']} | {a['outcomes']['expert_control']} | {a['outcomes']['left_open_once_recovery']} | {a['technical_complete_pair']} |" for a in audits)
    failures='\n'.join(f"- {a['seed']}："+json.dumps(a['errors'],ensure_ascii=False) for a in audits if a['errors']) or '无已记录异常。'
    (out/'report.md').write_text(f'''# 共享前缀恢复开发 v2：完整汇总

## 方法与数据

固定 seed 100123–100126，每个先独立生成 grasp/lift 前缀，再从相同初始化重放对照和开爪恢复。共12个计划进程、4个固定配对；前缀失败时两条件保持前置失败，不替换 seed。stride20、原控制器、原任务判据、CWMJ v2 资产不变。分支最多400动作/60000物理步/1200秒。

## 结果

| seed | 对照官方成功 | 恢复官方成功 | 完整技术配对 |
|---|---|---|---|
{table}

None 表示未知，不能改记失败或删除。对照成功 {control}/4，恢复成功 {recovery}/4，完整技术配对 {technical}/4。预定开发门：**{gate}**。训练准入：**False**。

![固定四对结果](matrix.png)

## 失败分析

{failures}

## 局限

任务成功与技术配对是不同条件；独立审计重新检查真实源文件、动作前缀、日志及可观测初始/边界状态。可观测状态仍不覆盖 PhysX 隐藏求解器缓存。此实验没有训练新模型，不能证明触觉模型收益。四对开发样本不作显著性检验，不能据此估计稳定恢复率。图像与 Strata 分析不能替代官方判据；不自动准入训练。

## 复现与下一步

协议 RECOVERY_DEVELOPMENT_PROTOCOL_V2_ZH.md；冻结清单和逐回合图文在 data/reports/recovery_development_v2，原始记录在 data/paired/recovery_v2_*。summary.json保存独立审计及来源摘要。依据技术失败类型修订下一份独立协议，或在门通过后另立规模化采集协议；不改本轮源记录、不重试原 seed、不覆盖 checkpoint、不自动启动 Strata。
''')
    return summary


if __name__=='__main__':report(Path(__file__).resolve().parents[1])

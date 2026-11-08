"""Full fixed-matrix pilot report after independent audit, never efficacy selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_feedback_pilot_final import audit,ARMS
from validate_expert_data import sha256


def report(root):
    root=Path(root);summary=audit(root)
    out=root/'data/reports/controlled_feedback_pilot_v1_final';out.mkdir(exist_ok=False)
    summary['analysis_sha256']=sha256(Path(__file__))
    summary['audit_sha256']=sha256(Path(__file__).with_name('audit_feedback_pilot_final.py'))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    matrix=np.full((4,4),np.nan);valid=np.zeros((4,4),dtype=bool)
    for row in summary['episodes']:
        i=(row['seed']-100127)*2+(0 if row['cadence']==10 else 1);j=ARMS.index(row['arm'])
        if type(row['success']) is bool:matrix[i,j]=float(row['success'])
        valid[i,j]=row['technical_pass']
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('lightgrey')
    ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set_xticks(range(4),ARMS);ax.set_yticks(range(4),[f'{s} / cadence {c}' for s in (100127,100128) for c in (10,1)])
    for i in range(4):
        for j in range(4):
            label='Unknown' if np.isnan(matrix[i,j]) else ('Success' if matrix[i,j] else 'Failure')
            if not valid[i,j]:label+='\nTechnical issue'
            ax.text(j,i,label,ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    ax.set_title('Technical pilot: fixed 16 attempts, not efficacy confirmation')
    fig.savefig(out/'matrix.png',dpi=150);plt.close(fig)
    lines=[]
    for row in summary['episodes']:
        name=f"s{row['seed']}_{row['arm']}_c{row['cadence']}"
        lines.append(f"| {row['seed']} | {row['cadence']} | {row['arm']} | {row['success']} | {row['technical_pass']} | {row.get('reason','—')} |")
    table='\n'.join(lines)
    (out/'report.md').write_text(f'''# 受控反馈技术试运行：固定16场汇总

## 方法与数据

两个预留开发seed，固定M8 r0/2880四组权重，每1或10动作重新观察。H50输出、10次去噪、原400动作/60000物理步/1200秒上限不变。噪声与错配供体按真实执行动作位置索引，统一没有额外影子推理。此试运行不构成正式收益确认，不训练新权重。

## 结果

技术通过门：**{summary['technical_gate']}**。固定分母16；跨频率初态/首命令配对：{summary['initial_first_command_pair_checks']}。

| seed | 反馈间隔 | 条件 | 官方成功 | 技术核验 | 异常或截断 |
|---|---|---|---|---|---|
{table}

![固定16场结果及技术状态](matrix.png)

## 失败分析与局限

None表示未知。资源截断或源数据异常保持技术问题，不因有成功字段就纳入可比结果。单场技术通过不替代八条件的初态配对门。初态和首命令必须相同，后续动作允许因重观察而不同。完整的真实反馈间隔与触觉年龄审计见summary.json和逐场报告。高频增加推理次数，墙钟延迟不等于仿真物理时间。

只有两个技术开发seed、一个训练重复，不报告显著性或可靠触觉收益。不按结果选择权重、反馈频率或成功seed。开发集不能成为独立确认集，自身策略输出不是专家监督，训练准入仍为false。未调用的Strata分析不能写成完成。

## 复现与下一步

协议CONTROLLED_FEEDBACK_PILOT_PROTOCOL_V1_ZH.md；冻结清单data/reports/controlled_feedback_pilot_v1/manifest.json；逐场图文在该目录reports；原始数据逻辑路径data/eval/controlled_feedback_pilot_v1。summary.json保存结果摘要、源审计、频率核验和分析源码SHA。技术门通过才另立多seed、三训练重复的确认协议；未通过则记录问题并用新协议修订，不重跑或覆盖本轮证据。
''')
    return summary


if __name__=='__main__':report(Path(__file__).resolve().parents[1])

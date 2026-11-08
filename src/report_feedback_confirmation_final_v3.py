"""Fixed 384-trial illustrated confirmation report including all unknowns."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_feedback_confirmation_v3 import audit,SEEDS,ARMS,identity
from validate_expert_data import sha256


def report(root):
    root=Path(root);summary=audit(root)
    out=root/'data/reports/feedback_confirmation_v3_final';out.mkdir(exist_ok=False)
    summary['reporter_sha256']=sha256(Path(__file__))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    matrix=np.full((16,8),np.nan);labels={}
    for i,s in enumerate(SEEDS):
        for ci,c in enumerate((10,1)):
            for ai,a in enumerate(ARMS):
                group=[r for r in summary['episodes'] if (r['seed'],r['cadence'],r['arm'])==(s,c,a)]
                if len(group)!=3:raise ValueError('three repeats per cell required')
                known=[r['success'] for r in group if r['technical_pass'] and type(r['success']) is bool]
                j=ci*4+ai;labels[i,j]=f'{sum(known)}/3' if len(known)==3 else f'{sum(known)} S\n{3-len(known)} ?'
                if len(known)==3:matrix[i,j]=sum(known)/3
    fig,ax=plt.subplots(figsize=(12,10),layout='constrained');cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('lightgrey')
    ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set_xticks(range(8),[f'c{c}\n{a}' for c in (10,1) for a in ARMS]);ax.set_yticks(range(16),list(map(str,SEEDS)))
    for (i,j),label in labels.items():ax.text(j,i,label,ha='center',va='center',fontsize=8)
    ax.set_title('Successes / 3 fixed trained repeats; ? = unknown (not failure)')
    fig.savefig(out/'matrix.png',dpi=150);plt.close(fig)
    stat=summary['statistics'];contrasts=[]
    if stat is not None:
        for x in stat['contrasts']:
            contrasts.append(f"| {x['name']} | {x['effect']:.4f} | {x['seed_bootstrap_95']} | {x['holm_p']:.6g} | {x['gate']} |")
    table='\n'.join(contrasts) if contrasts else '| 未计算：技术证据不完整 | — | — | — | False |'
    episodes=[]
    for r in summary['episodes']:
        name=identity(r['seed'],r['arm'],r['repeat'],r['cadence'])
        relative=f'../feedback_confirmation_v3/reports/{name}/report.md'
        exists=(out/relative).is_file()
        episodes.append(f"| {r['seed']} | {r['repeat']} | {r['cadence']} | {r['arm']} | {r['success']} | {r['technical_pass']} | {r.get('error','—').replace('|','/')} | "+(f'[图文]({relative})' if exists else '报告缺失')+' |')
    missing=sum('报告缺失' in x for x in episodes)
    (out/'report.md').write_text(f'''# 受控反馈正式确认：完整384场报告

## 方法与数据

16个独立环境seed，三个既有M8训练重复各自2880步权重，四条件，间隔10/1。固定原任务判据、H50、10次去噪、400动作与统一物理/墙钟上限。统计单位是seed，三个训练重复在seed内平均；无新训练数据混入。全部原始记录来自本机仿真，不代表真实硬件。

## 技术证据与结果

完整技术门：**{summary['technical_gate']}**；固定回合数384；未知回合：**{summary['unknown_trials']}**；逐场图文缺失：**{missing}**。单元格仅汇总三个训练重复，不代表三个独立环境seed；灰色含未知值。48组初态/首命令配对详情见summary.json。

![全部确认场景](matrix.png)

| 预定对比 | 效应 | seed bootstrap 95%区间 | Holm p | 正收益门 |
|---|---:|---|---:|---|
{table}

高频独有触觉收益：{stat['high_frequency_unique_tactile_benefit'] if stat else '未判定'}；反馈频率交互支持：{stat['feedback_interaction_supported'] if stat else '未判定'}。

## 失败分析与局限

普通任务失败计0；资源截断、源数据异常或运行错误保留unknown，不能填0。技术证据不全时不输出主要确认统计。全部任务失败仍可构成有效负结果，不以正结果为完成条件。非显著不等于等效或触觉无用；符号翻转依赖差值对称假设，bootstrap条件于三套固定权重，16个seed的统计功效有限。图像最后请求帧不等于动作后终态，触觉proxy未作硬件标定。Strata辅助分析未调用时不视为完成，也不决定官方成功标签。

## 逐场完整记录

| seed | repeat | cadence | 条件 | 审计后成功 | 技术通过 | 异常 | 报告 |
|---|---|---|---|---|---|---|---|
{chr(10).join(episodes)}

## 复现与下一步

正式协议CONTROLLED_FEEDBACK_CONFIRMATION_PROTOCOL_V3_ZH.md；冻结清单data/reports/feedback_confirmation_v3/manifest.json。summary.json包含384条源审计、48组配对及固定统计。逐场图文链接如上；缺失的报告必须补齐后才能称交付完整。若技术门不通过，优先诊断技术异常并另立协议，不重跑或覆盖已消耗seed；若通过，按预定对比报告正/负/不确定结论，再决定数据采集或模型修改，不事后更换权重或seed以追求正结果。
''')
    return summary

if __name__=='__main__':report(Path(__file__).resolve().parents[1])

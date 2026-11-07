"""Complete inventory, technical gates, then predeclared seed-level analysis."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m8_completion_inventory import inventory,planned_attempts,CONDITIONS
from m8_paired_seed_statistics import summarize
from audit_cwmj_m8_pair import audit_pair,audit_initial_across_nodes_and_repeats
from verify_cwmj_m8_confirmation import verify_confirmation
from validate_expert_data import sha256

R=Path(__file__).resolve().parents[1]


def report(root=R, destination=None):
    state=inventory(root)
    if not state['ready_for_final_analysis']:raise RuntimeError('pending attempts; no final analysis')
    eligible=[];values=[];excluded=[];pair_audits={};all_attempts=[]
    for name,(seed,r,steps,arm) in planned_attempts().items():
        verify_confirmation(root,seed,r,arm,steps)
        if name in state['terminal']:
            raw=state['terminal'][name]
            if raw['status']=='completed' and raw['confirmation_manifest_sha256']!=sha256(root/'data/reports/cwmj_m8/confirmation_manifest.json'):
                raise ValueError('result confirmation provenance changed')
            all_attempts.append(dict(name=name,result=raw,result_sha256=sha256(root/f'data/eval/cwmj_m8_confirm_{name}/result.json')))
    for seed in range(100103,100119):
        if seed in state['interrupted_seeds']:
            excluded.append(dict(seed=seed,reason='audited interruption; whole seed excluded'));continue
        cube=np.zeros((3,2,4),dtype=int);paths=[];valid=True
        for r in range(3):
            for si,steps in enumerate((720,2880)):
                trials={arm:root/f'data/eval/cwmj_m8_confirm_r{r}_s{steps}_{seed}_{arm}' for arm in CONDITIONS}
                audit=audit_pair(trials,seed,r,steps)
                pair_audits[f'{seed}_r{r}_s{steps}']=audit
                valid=valid and audit.get('eligible_complete_pair',False)
                paths.extend(trials.values())
                for ai,arm in enumerate(CONDITIONS):
                    cube[r,si,ai]=int(audit['outcomes'][arm] is True)
        if valid:valid=audit_initial_across_nodes_and_repeats(paths)
        if not valid:excluded.append(dict(seed=seed,reason='incomplete or technical pairing gate failed'));continue
        eligible.append(seed);values.append(cube)
    x=np.asarray(values,dtype=int).reshape(len(eligible),3,2,4)
    summary=summarize(x,eligible)
    summary.update(exclusions=excluded,planned_attempts=384,terminal_count=len(state['terminal']),
                   interruptions=state['interrupted'],pair_audits=pair_audits,all_attempts=all_attempts,
                   analysis_code_sha256=sha256(root/'src/m8_paired_seed_statistics.py'))
    out=root/'data/reports/cwmj_m8_confirmation_summary';out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    matrix=np.full((16,24),np.nan)
    for name,(seed,r,steps,arm) in planned_attempts().items():
        row=state['terminal'].get(name)
        if row and row['status']=='completed':matrix[seed-100103,r*8+(0 if steps==720 else 4)+CONDITIONS.index(arm)]=row['success']
    fig,ax=plt.subplots(figsize=(15,7),layout='constrained')
    cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('lightgrey')
    ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set_yticks(range(16),[str(s) for s in range(100103,100119)])
    ax.set_xticks(range(24),[f'r{r} / {s} / {a}' for r in range(3) for s in (720,2880) for a in CONDITIONS],rotation=75,ha='right')
    ax.set_title('All attempted episodes: green success, red failure, grey unavailable')
    fig.savefig(out/'matrix.png',dpi=150);plt.close(fig)
    rows='\n'.join(f"| {c['name']} | {c['effect']} | {c['seed_bootstrap_95']} | {c['holm_p']:.6f} | {c['gate']} |" for c in summary['contrasts'])
    (out/'report.md').write_text(f'''# M8 固定数据训练时长：闭环图文汇报

## 方法与数据

16个预留新seed、三个训练重复、720/2880两个固定节点、四条件，共384个计划回合。训练数据、模型容量、反馈频率和任务判据保持一致；1440只作离线曲线，未按验证成绩选择节点。独立统计单位为seed，三个训练重复先平均，不能把384回合当独立样本。

## 结果

终态记录{len(state['terminal'])}，审计中断{len(state['interrupted'])}，完整技术配对seed {len(eligible)}/16。训练不足解释获得支持：**{summary['duration_explanation_supported']}**；2880步独有触觉收益：**{summary['final_unique_tactile_benefit']}**。

![全部回合矩阵](matrix.png)

| 对比 | 成功率差 | seed簇95%区间 | Holm p | 判据 |
|---|---:|---|---:|---|
{rows}

## 失败分析与局限

排除记录：{json.dumps(excluded,ensure_ascii=False)}。缺失或初始化异常没有伪造为失败，整seed排除可能引入选择偏差。未达到判据不能证明等效或触觉无效。错配保留接触分层信息；本试验只检验固定数据下训练时长，不能说明增加恢复示范或反馈频率的效果。图中终态依据仿真原判据，Strata仅作辅助解释。

## 复现与下一步

冻结协议 CWMJ_M8_FROZEN_PROTOCOL_ZH.md；训练与闭环清单位于data/reports/cwmj_m8。summary.json保存每回合原结果、SHA、配对检查、排除和统计。原始数据在data/eval/cwmj_m8_confirm_*，逐回合图文在data/reports/cwmj_m8_confirmation。根据三组对比及全部失败轨迹决定后续数据扩充和恢复监督实验，另立协议，不回改本实验。
''')
    dest=Path(destination) if destination is not None else Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m8-confirmation')
    dest.mkdir(exist_ok=True)
    for p in out.iterdir():
        if (dest/p.name).exists():raise ValueError('published summary already exists')
        shutil.copy2(p,dest/p.name)


if __name__=='__main__':report()

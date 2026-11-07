"""Fixed eight-attempt feasibility report, not learned tactile benefit evidence."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_recovery_pair import audit
from validate_expert_data import sha256

R=Path(__file__).resolve().parents[1]
SEEDS=tuple(range(100119,100123))
CONDITIONS=('expert_control','left_open_once_recovery')


def feasibility(records,pairs):
    expected={(s,c) for s in SEEDS for c in CONDITIONS}
    if set(records)!=expected or set(pairs)!=set(SEEDS):raise ValueError('all eight attempts and four pair audits required')
    unknown=[];control=0;recovered=0
    for (seed,condition),row in records.items():
        if row.get('seed')!=seed or row.get('condition')!=condition:raise ValueError('attempt identity mismatch')
        if row.get('status')!='completed':unknown.append([seed,condition]);continue
        if type(row.get('official_success')) is not bool:raise ValueError('explicit official outcome required')
        if condition=='expert_control':control+=int(row['official_success'] and row.get('termination')=='task_success')
        elif (row['official_success'] and row.get('intervention_applied') is True
              and row.get('usable_recovery_candidate') is True):recovered+=1
    valid_pairs=[s for s in SEEDS if pairs[s].get('technical_comparable') is True]
    # Every completed pair must pass initial/prefix checks. Unknown attempts
    # prohibit the gate; they are neither dropped nor counted as task failures.
    passed=not unknown and control>=3 and recovered>=2 and len(valid_pairs)==4
    return dict(control_successes=control,usable_recovery_successes=recovered,denominator_per_condition=4,
                unknown_attempts=unknown,technical_complete_pairs=valid_pairs,feasibility_gate=passed,
                learned_tactile_benefit_proven=False,training_approval=False)


def report(root=R,destination=None):
    records={};paths={};source_hashes={}
    for seed in SEEDS:
        for condition in CONDITIONS:
            trial=root/f'data/paired/recovery_dev_{seed}_{condition}';paths[seed,condition]=trial
            result=trial/'result.json';failure=trial/'failure.json'
            if not result.exists() and not failure.exists():raise RuntimeError('unattempted/live development trial; no final summary')
            if result.exists() and failure.exists():raise ValueError('ambiguous success/failure records require inspection')
            selected=result if result.exists() else failure
            records[seed,condition]=json.loads(selected.read_text());source_hashes[str(selected)]=sha256(selected)
    pairs={s:audit(paths[s,CONDITIONS[0]],paths[s,CONDITIONS[1]],s) for s in SEEDS}
    summary=feasibility(records,pairs)
    summary.update(pair_audits=pairs,source_sha256=source_hashes,all_attempts=list(records.values()))
    out=root/'data/reports/recovery_development_summary_v1';out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    matrix=np.full((4,2),np.nan)
    for i,s in enumerate(SEEDS):
        for j,c in enumerate(CONDITIONS):
            row=records[s,c]
            if row.get('status')=='completed':matrix[i,j]=row['official_success']
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('lightgrey')
    ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set_xticks([0,1],['Expert control','One left-open recovery']);ax.set_yticks(range(4),[str(s) for s in SEEDS])
    ax.set_title('Official outcomes; grey = unknown')
    fig.savefig(out/'matrix.png',dpi=150);plt.close(fig)
    rows='\n'.join(f"| {s} | {records[s,CONDITIONS[0]].get('official_success','未知')} | {records[s,CONDITIONS[1]].get('official_success','未知')} | {pairs[s].get('technical_comparable',False)} |" for s in SEEDS)
    (out/'report.md').write_text(f'''# 固定开爪恢复开发：完整汇总

固定四个开发seed、两条件共八回合，不补选seed、不进行显著性检验。原控制器、CWMJ v2、原成功判据、400动作/60000物理步/1200秒上限保持一致。扰动条件在lift后打开一次左爪并从当前场景重抓；干预不能当专家标签。

![全部官方终态](matrix.png)

| seed | 无扰动官方成功 | 扰动官方成功 | 技术配对通过 |
|---|---|---|---|
{rows}

无扰动成功{summary['control_successes']}/4；实际触发、成功且数据合格的恢复候选{summary['usable_recovery_successes']}/4。未知：{summary['unknown_attempts']}。预定开发可行性门：**{summary['feasibility_gate']}**。

未知没有被删除或改成任务失败；完整回合还必须通过初态、干预前动作/时钟、资产与原始文件SHA审计。失败和异常详情保留在summary.json及逐回合报告中，不能仅按矩阵颜色判断失败原因。

这只检验规则专家与数据采集可行性，不代表学习后模型的恢复能力或触觉收益。开发样本少，不能当独立确认集；训练授权仍为False。通过后另立规模化训练/验证采集协议，未通过则按固定回合证据修订并使用新seed，不改写本实验。

复现：RECOVERY_DEVELOPMENT_PROTOCOL_V1_ZH.md、冻结实施清单及summary.json中的源SHA。每回合原始数据位于data/paired/recovery_dev_*，图文位于data/reports/recovery_development。下一步将数据审计与正式训练准入分开，不用Strata生成文字替代官方终态。
''')
    dest=Path(destination) if destination is not None else Path('/home/admin001/lingbot-tactile-artifacts/recovery-development')
    dest.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if (dest/p.name).exists():raise ValueError('summary already published')
        shutil.copy2(p,dest/p.name)


if __name__=='__main__':report()

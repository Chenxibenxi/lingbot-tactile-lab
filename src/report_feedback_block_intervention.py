"""Whole-batch-only analysis; never reports partial success rates."""
import json
from pathlib import Path
import numpy as np
import h5py
from audit_feedback_block_trace import audit
from audit_feedback_block_input import audit_input
from validate_expert_data import sha256

R=Path(__file__).resolve().parents[1]
NAMES=[f's{s}_{a}_{m}' for s in (100145,100146) for a in ('live','history_zero') for m in ('fresh_first','cached_block')]

def require_complete(root):
    receipts=root/'journal_remaining/receipts'
    if {p.stem for p in receipts.glob('*.json')}!=set(NAMES):
        raise ValueError('whole eight-trial batch required; no partial analysis')
    for name in NAMES:
        v=json.loads((receipts/(name+'.json')).read_text())
        if v.get('status')!='completed' or v.get('returncode')!=0 or v.get('resource_censored') is not False:
            raise ValueError('technical failure/censoring requires separate failure report')

def main():
    reports=R/'data/reports/feedback_block_intervention_v1'
    require_complete(reports)
    rows=[];initial={}
    for name in NAMES:
        trial=R/'data/eval/feedback_block_intervention_v1'/name
        v=json.loads((trial/'result.json').read_text())
        proof=audit(trial/'block_trace.jsonl',trial/'rollout.hdf5');input_proof=audit_input(trial)
        trace=[json.loads(s) for s in (trial/'block_trace.jsonl').read_text().splitlines()]
        actions=[x for x in trace if x['event']=='action']
        errors=[]
        for x in actions:
            differences=[]
            for side in ('left','right'):
                q=np.asarray(x['after'][side]['arm_actual_qpos']);t=np.asarray(x['after'][side]['arm_drive_targets'])
                if q.shape!=(6,) or t.shape!=(6,) or not np.isfinite(q).all() or not np.isfinite(t).all():raise ValueError('physical state')
                differences.extend(abs(q-t))
            errors.append(np.mean(differences))
        raw=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.policy.jsonl','.frames.jsonl'))]
        if len(raw)!=1:raise ValueError('unique raw stream')
        first={}
        with raw[0].open() as f:
            next(f)
            for line in f:
                x=json.loads(line)
                for side in ('left','right'):
                    if x[side]['load_proxy_n']>0:
                        for obj in x[side]['objects']:first.setdefault(obj,x['physics_step'])
        windows={}
        for obj,step in first.items():
            hit=[i for i,x in enumerate(actions) if x['physics_before']<step<=x['physics_after']]
            if len(hit)!=1:raise ValueError('contact action mapping')
            start=hit[0]+1
            windows[obj]=dict(first_physics_step=step,containing_action=hit[0],post20_mae=float(np.mean(errors[start:start+20])) if len(errors)-start>=20 else None)
        with h5py.File(trial/'rollout.hdf5') as h:
            initial[name]={k:h[k][0] for k in ('images/head','images/left','images/right','state_14','applied/command_14')}
        initial[name]['scene']=json.dumps(v['initial_scene'],sort_keys=True)
        rows.append(dict(name=name,success=v['success'],actions=v['actions'],wall_seconds=v['wall_seconds'],first60_mae=float(np.mean(errors[:60])) if len(errors)>=60 else None,contact_windows=windows,source_audit=proof,input_audit=input_proof,result_sha256=sha256(trial/'result.json')))
    pairs={}
    for seed in (100145,100146):
        names=[n for n in NAMES if n.startswith(f's{seed}_')];ref=initial[names[0]]
        checks={n:{k:bool(np.array_equal(ref[k],initial[n][k])) for k in ref} for n in names[1:]}
        if not all(all(c.values()) for c in checks.values()):raise ValueError('initial pairing mismatch')
        pairs[str(seed)]=checks
    out=R/'data/reports/feedback_block_intervention_summary_v1';out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,initial_pairing=pairs),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    ax.bar(range(8),[x['first60_mae'] if x['first60_mae'] is not None else np.nan for x in rows])
    ax.set_xticks(range(8),[x['name'].replace('s100','') for x in rows],rotation=55,ha='right');ax.set_ylabel('First 60 actions: actual qpos vs drive target MAE (rad)')
    fig.savefig(out/'first60.png',dpi=150);plt.close(fig)
    table='\n'.join(f"|{x['name']}|{x['success']}|{x['actions']}|{x['first60_mae']}|{x['wall_seconds']:.1f}|" for x in rows)
    (out/'report.md').write_text('''# 动作块执行来源干预：8场开发机制实验

## 方法与数据
固定seed100145/146及同一r0/2880 true适配器；live/历史置零×fresh_first/cached_block。两组每步观察推理；前者执行新预测首动作，后者每10步保存预测并顺次执行。保持原成功判据，独立来源审计核对全部实际命令；不是独立确认集。

![前60动作误差](first60.png)

|场次|成功|动作数|前60动作MAE(rad)|运行秒数|
|---|---|---:|---:|---:|
'''+table+'''

## 失败分析与局限
以上是完整矩阵的描述性结果，不进行两个开发seed的显著性判断。缓存条件延迟新观察进入执行，不能将其效果单独归因触觉。首次接触按物理步匹配动作，后20完整动作和缺失值见summary.json。低跟踪误差不保证策略正确；结果不批准策略轨迹为专家训练标签。

## 复现与下一步
入口src/run_feedback_block_intervention_batch.py；冻结文件data/reports/feedback_block_intervention_v1/manifest.json；汇总src/report_feedback_block_intervention.py。逐场相机与接触图见原批次reports目录。需人工检查图像，并比较缓存轨迹与旧c10的额外观察副作用，再决定独立seed验证或无接触残差干预。报告自动生成不代表机制归因审查完成。未调用Strata。
''')
    print(out)

if __name__=='__main__':main()

"""Retrospective sensor observation coverage; no model inputs or labels are changed."""
from pathlib import Path
import json,shutil
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]

def coverage(contact,anchors,valid,window_steps=25):
    contact=np.asarray(contact,dtype=bool);anchors=np.asarray(anchors,dtype=int);valid=np.asarray(valid,dtype=bool)
    if contact.ndim!=1 or anchors.shape!=valid.shape or np.any(anchors<0) or np.any(anchors>len(contact)) or np.any(np.diff(anchors)<0):raise ValueError('invalid physical clock')
    observed=np.zeros(len(contact),dtype=bool)
    for a,v in zip(anchors,valid):
        if v:
            if a<window_steps:raise ValueError('valid window lacks physical history')
            observed[a-window_steps:a]=True # array index0 is physics step1, includes anchor
    last=int(anchors[-1]) if len(anchors) else 0
    eligible=contact[:last];seen=observed[:last];changes=np.diff(np.r_[False,eligible,False].astype(int));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
    return {'all_contact_steps':int(contact.sum()),'contact_steps_before_last_request':int(eligible.sum()),'contact_steps_in_query_windows':int((eligible&seen).sum()),'contact_step_capture_fraction':float((eligible&seen).sum()/eligible.sum()) if eligible.any() else None,'contact_runs_before_last_request':len(starts),'contact_runs_seen_by_any_window':sum(bool(seen[s:e].any()) for s,e in zip(starts,ends)),'tail_contact_steps_after_last_request':int(contact[last:].sum()),'observed_physical_steps':int(observed.sum()),'total_physical_steps':len(contact)}

def report():
    out=R/'data/reports/contact_cadence_m4_m5';out.mkdir(exist_ok=True);rows=[]
    fig,axes=plt.subplots(3,4,figsize=(13,7),layout='constrained')
    for i,(stage,arm,label) in enumerate([('m4','zero','Zero-trained / zero-input'),('m4','true','True-trained / true-input'),('m5','zero','True-trained / zero-input')]):
        for j,seed in enumerate(range(100051,100055)):
            trial=R/f'data/eval/cwmj_{stage}_{seed}_{arm}';result=json.loads((trial/'result.json').read_text());raws=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.policy.jsonl','.frames.jsonl'))]
            if len(raws)!=1:raise ValueError('unique raw source required')
            raw=raws[0];hpath=trial/'rollout.hdf5';audit=result['source_audit']['sha256']
            if sha256(raw)!=audit['raw'] or sha256(hpath)!=audit['hdf5']:raise ValueError('rollout source drift')
            with raw.open() as f:
                head=json.loads(next(f));contact=[]
                for step,line in enumerate(f,1):
                    r=json.loads(line)
                    if r['physics_step']!=step:raise ValueError('noncontiguous raw clock')
                    contact.append(any(float(v['load_proxy_n'])>0 for side in ['left','right'] for v in r[side]['fingers'].values()))
            with h5py.File(hpath) as h:anchors=h['request/physics_step'][:];valid=h['tactile/window_valid'][:,0].astype(bool)
            d=coverage(contact,anchors,valid);dt=head['dt_s'];gaps=np.diff(anchors)*dt
            d.update(seed=seed,condition=label,median_query_gap_s=float(np.median(gaps)) if len(gaps) else None,max_query_gap_s=float(gaps.max()) if len(gaps) else None,window_s=25*dt,success=result['success'],raw_sha256=audit['raw'],hdf5_sha256=audit['hdf5']);rows.append(d)
            ax=axes[i,j];c=np.array(contact);ax.plot((np.arange(len(c))+1)*dt,c.astype(int),lw=.5,color='#829ba5',label='Any finger load > 0')
            for a,v in zip(anchors,valid):
                if v:ax.axvspan((a-25)*dt,a*dt,color='#c74d35',alpha=.6)
            ax.set_title(f'{seed}: sampled {100*d["contact_step_capture_fraction"]:.1f}%' if d['contact_step_capture_fraction'] is not None else f'{seed}: no contact',fontsize=9);ax.set_ylim(-.1,1.2);ax.set_xlabel('Simulated seconds',fontsize=8)
            if j==0:ax.set_ylabel(label+'\nContact present',fontsize=8)
    fig.suptitle('Recorded contact vs 100 ms query windows (red) | retrospective diagnostic');fig.savefig(out/'contact_timeline.png',dpi=160);plt.close(fig)
    summary={'schema':'retrospective_contact_sampling_audit_v1','window_steps':25,'cadence_actions':10,'not_causal_performance_evidence':True,'rows':rows};(out/'summary.json').write_text(json.dumps(summary,indent=2))
    table='\n'.join(f"| {d['seed']} | {d['condition']} | {d['median_query_gap_s']:.3f} | {d['contact_step_capture_fraction']:.1%} | {d['contact_runs_seen_by_any_window']}/{d['contact_runs_before_last_request']} | {d['tail_contact_steps_after_last_request']} |" for d in rows)
    text=f'''# M4/M5触觉观测频率审计：完整图文报告

## 目的与方法

离线只读全部12个策略回合，重新核验raw/HDF5 SHA。沿用接口定义：任一手指load_proxy_n>0为接触，不额外调阈值。模型请求仅输入当时及之前25物理步（约100ms）；每10动作重规划，但每动作经过的物理时间并不恒定。

统计最后一次请求之前所有接触物理步中，落入任一有效请求窗口的比例。最后请求之后的接触单列，避免终止尾段混淆。连续接触段以每物理步>0的连续区间定义，可能含极短接触与数值抖动，不能直接称为抓取事件或滑移。无力/消融行表示传感器可用历史覆盖，实际模型输入已置零。

![接触与实际请求窗口](contact_timeline.png)

| seed | 条件 | 请求间隔中位数秒 | 接触步被窗口覆盖 | 被覆盖连续段/全部段 | 最后请求后接触步 |
|---|---|---:|---:|---:|---:|
{table}

## 解释与局限

窗口覆盖比例不是成功率，不要求每个持续接触物理步都被模型看到；只需少量代表性观测也可能足够。较低覆盖提示当前反馈频率/历史长度值得独立检验，但本分析不能证明它导致失败。多段接触可能是同一抓取的细碎信号；接触有无不能判断抓稳、滑落或对象受损。

M5同权重输入消融成功率与真实输入相同，唯一共同成功回合动作数不同；M4不同权重又在接触前分叉。两者均要求进一步独立数据和对照，不能以本审计替代任务效益验证。

## 下一步和复现

保持M6已冻结采集接口，先扩充独立数据。未来另立更短重规划间隔/更长历史/接触摘要的公平对照：更短动作块也改变视觉反馈频率，必须比较相同频率下的zero/true；更长历史须重新定义时间契约及训练数据，不能只在推理时悄悄替换。增加瓶功能点时序记录才能可靠定位抬升/放置失败阶段。

代码 `{R}/src/audit_contact_observation_cadence.py`；结果summary.json附原数据SHA。此诊断只用CPU，不更新模型、不回流rollout标签、不启动仿真、不操作Strata。
'''
    (out/'report.md').write_text(text);dest=Path('/home/admin001/lingbot-tactile-artifacts/contact-cadence-m4-m5');shutil.copytree(out,dest,dirs_exist_ok=True)
    print(json.dumps([{'seed':d['seed'],'condition':d['condition'],'capture':d['contact_step_capture_fraction'],'median_gap_s':d['median_query_gap_s']} for d in rows]),flush=True)
if __name__=='__main__':report()

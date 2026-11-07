"""Illustrated per-arm rollout report, plus paired equality audit when ready."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from validate_expert_data import read_jsonl
R=Path(__file__).resolve().parents[1]
DEST=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m5-closed-loop')

def report(seed,arm):
    trial=R/f'data/eval/cwmj_m5_{seed}_{arm}'
    r=json.loads((trial/'result.json').read_text());out=R/f'data/reports/cwmj_m5_closed_loop/{seed}_{arm}';out.mkdir(parents=True,exist_ok=False)
    raw=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.policy.jsonl','.frames.jsonl'))]
    rows=read_jsonl(raw[0])[1:] if len(raw)==1 else []
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');image_text=''
    if (trial/'rollout.hdf5').exists():
        with h5py.File(trial/'rollout.hdf5') as h:
            commands=h['applied/command_14'][:]
            if len(commands):
                axes[0].plot(commands[:,6],label='left gripper command');axes[0].plot(commands[:,13],label='right gripper command');axes[0].legend(fontsize=8)
            n=len(h['state_14']);contact_idx=next((q['request_index'] for q in r['requests'] if q['past_contact']),max(n-1,0))
            if n:
                f,axs=plt.subplots(1,3,figsize=(10,3),layout='constrained')
                for ax,i,label in zip(axs,[0,contact_idx,n-1],['Initial','First past-contact request','Last captured pre-action']):ax.imshow(h['images/head'][i]);ax.set_title(f'{label} ({i})',fontsize=8);ax.axis('off')
                f.savefig(out/'camera_frames.png',dpi=160);plt.close(f);image_text='![闭环真实相机帧](camera_frames.png)'
    axes[0].set(xlabel='Actual action index',ylabel='Gripper command (0–1)',title='Executed commands')
    if rows:
        for side in ['left','right']:axes[1].plot([x['physics_step'] for x in rows],[x[side]['load_proxy_n'] for x in rows],label=side,lw=.6)
        axes[1].set_yscale('symlog',linthresh=.1);axes[1].legend()
    else:axes[1].text(.5,.5,'No recorded physical actions',ha='center',transform=axes[1].transAxes)
    axes[1].set(xlabel='Recorded physical step',ylabel='Impulse/dt proxy (N)',title='Live tactile feedback')
    fig.savefig(out/'rollout_trace.png',dpi=160);plt.close(fig)
    summary={k:v for k,v in r.items() if k not in ['requests','source_audit','source_evidence']}
    summary['contact_requests']=sum(q['past_contact'] for q in r['requests']);summary['total_requests']=len(r['requests']);summary['shadow_zero_input']=[q for q in r['requests'] if 'zero_input_max_action_delta' in q]
    (out/'result.json').write_text(json.dumps(summary,indent=2))
    text=f'''# M5 同true权重、输入置零闭环实验：seed{seed} / {arm}

## 协议与方法

本次重用M4测试seed51–54，属于结果启发的探索性机制消融，不是独立确认实验。固定true组最后360步权重，推理history置零，保留mask/年龄/有效性；对照为M4同seed的true输入回合。原400真实动作/每10步重规划/10步去噪，原pick_diverse_bottles成功判据。场景是CWMJ单酒瓶双侧，不能与原20瓶分布混算。原传感器只记录过去接触；实际输入history全部置零，原HDF5记录的是物理观测，不是消融后模型输入。权重SHA及protocol SHA见结果。

## 结果

| 项目 | 结果 |
|---|---|
| 完成状态 | {r['status']} |
| 原任务成功 | {r.get('success','未产生可评分结果')} |
| 终止原因 | {r.get('termination',r.get('exception'))} |
| 真实执行动作 | {r['actions']} |
| 记录物理步 | {r['physics_steps']} |
| 过去接触请求/总请求 | {summary['contact_requests']} / {summary['total_requests']} |

![实际执行与触觉](rollout_trace.png)

{image_text}

## 分析与局限

当前是一个arm、一个seed，不能单独估计触觉因果收益；同seed配对完成后另作逐元素初始状态/动作一致性审计。初始化异常、推理异常和资源截断分别标识，不记为完整任务失败。最后相机帧发生在动作前，最终成功以原任务判据及episode_outcome为准。失败轨迹不能直接当专家监督。

```json
{json.dumps(summary,ensure_ascii=False,indent=2)}
```

## 复现与下一步

原始HDF5/触觉物理时钟/动作及独立校验在 `{trial}`；权重SHA、原源码证据见result.json，资产工厂/manifest见scenario.json。图中力是仿真代理量，未作真实传感器校准。继续完成全部固定配对后报告整体方向与局限，不按成绩补换seed。每个进程释放自身GPU租约，不自动重启Strata。
'''
    (out/'report.md').write_text(text);DEST.mkdir(exist_ok=True);shutil.copytree(out,DEST/out.name)
    a=R/f'data/eval/cwmj_m5_{seed}_zero';b=R/f'data/eval/cwmj_m4_{seed}_true'
    if (a/'result.json').exists() and (b/'result.json').exists():
        pair={'exploratory_reused_test_seeds':True,'seed':seed,'outcomes':{arm:json.loads((p/'result.json').read_text()).get('success') for arm,p in [('zero',a),('true',b)]}}
        ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text())
        pair['same_adapter_sha256']=ra.get('adapter_sha256')==rb.get('adapter_sha256') and bool(ra.get('adapter_sha256'))
        pair['same_scenario']=json.loads((a/'scenario.json').read_text())==json.loads((b/'scenario.json').read_text())
        if not pair['same_adapter_sha256'] or not pair['same_scenario']:raise ValueError('same-weight or asset gate failed')
        if (a/'rollout.hdf5').exists() and (b/'rollout.hdf5').exists():
            with h5py.File(a/'rollout.hdf5') as x,h5py.File(b/'rollout.hdf5') as y:
                if len(x['state_14']) and len(y['state_14']):
                    first_contact=next((q['request_index'] for q in rb['requests'] if q['past_contact']),len(rb['requests']))
                    prefix=min(first_contact*10,len(y['applied/command_14']))
                    pair['before_first_contact_actions_equal']=np.array_equal(x['applied/command_14'][:prefix],y['applied/command_14'][:prefix])
                    if not pair['before_first_contact_actions_equal']:raise ValueError('same-weight trajectories diverged before tactile input difference')
                    pair.update(initial_rgb_and_state_equal=all(np.array_equal(x[k][0],y[k][0]) for k in ['images/head','images/left','images/right','state_14']),
                                first10_actions_equal=np.array_equal(x['applied/command_14'][:10],y['applied/command_14'][:10]),
                                invalid_tactile_first_both=not bool(x['tactile/window_valid'][0,0]) and not bool(y['tactile/window_valid'][0,0]))
        p=out.parent/f'paired_{seed}.json';p.write_text(json.dumps(pair,indent=2));shutil.copy2(p,DEST/p.name)
        if 'initial_rgb_and_state_equal' in pair and not all(pair[k] for k in ['initial_rgb_and_state_equal','first10_actions_equal','invalid_tactile_first_both']):raise ValueError('paired initial reproducibility failed; stop comparison')
    print(json.dumps({'seed':seed,'arm':arm,'status':r['status'],'success':r.get('success'),'actions':r['actions']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['zero','true'],required=True);a=p.parse_args();report(a.seed,a.arm)

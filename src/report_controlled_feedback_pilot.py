"""Per-attempt pilot report; no confirmation claims or invented missing images."""
import json
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_expert_data import sha256,read_jsonl


def report(trial,out):
    trial,out=Path(trial),Path(out)
    result=json.loads((trial/'result.json').read_text())
    if result.get('status') not in ('completed','failed'):raise ValueError('terminal result required')
    out.mkdir(parents=True,exist_ok=False)
    requests=result.get('requests',[])
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    axes[0].plot([r['executed_actions'] for r in requests],[r['capture_physics_step'] for r in requests],'.-',lw=.7)
    axes[0].set(title='Actual observation boundaries',xlabel='Executed actions',ylabel='Physical step')
    ages=[t['history_age_at_action_end_s'] for r in requests for t in r.get('executed_action_timing',[])]
    axes[1].plot(ages,lw=.7);axes[1].set(title='History age at action end',xlabel='Executed action index',ylabel='Simulation seconds')
    raw=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.policy.jsonl','.frames.jsonl'))]
    if len(raw)==1:
        physics=read_jsonl(raw[0])[1:]
        for side in ('left','right'):
            axes[2].plot([x['physics_step'] for x in physics],[x[side]['load_proxy_n'] for x in physics],label=side,lw=.5)
        axes[2].set_yscale('symlog',linthresh=.1);axes[2].legend()
    else:axes[2].text(.5,.5,'No unique physical stream',ha='center',transform=axes[2].transAxes)
    axes[2].set(title='Recorded tactile proxy',xlabel='Physical step',ylabel='Uncalibrated impulse/dt')
    fig.savefig(out/'timing_contact.png',dpi=140);plt.close(fig)
    image_note='没有可读取的相机记录，不补造画面。'
    hdf5=trial/'rollout.hdf5'
    if hdf5.exists():
        try:
            with h5py.File(hdf5,'r') as h:
                frames=h['images/head'];n=len(frames)
                if n:
                    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
                    for ax,index,label in zip(axes,(0,n-1),('Initial request','Last captured pre-action')):
                        ax.imshow(frames[index]);ax.set_title(f'{label} ({index})');ax.axis('off')
                    fig.savefig(out/'camera_frames.png',dpi=140);plt.close(fig)
                    image_note='![真实相机请求帧](camera_frames.png)'
        except (OSError,KeyError,ValueError) as exc:image_note=f'相机记录不可读：{type(exc).__name__}: {exc}'
    evidence={str(p):sha256(p) for p in (trial/'result.json',hdf5) if p.is_file()}
    (out/'summary.json').write_text(json.dumps(dict(result=result,evidence_sha256=evidence,
        confirmation_eligible=False,training_approval=False),ensure_ascii=False,indent=2))
    (out/'report.md').write_text(f'''# 受控反馈技术试运行：{trial.name}

## 方法与数据

固定两个技术seed、四条件、反馈间隔1/10，共16回合；只用M8 r0/2880固定权重。按已执行动作位置配对噪声和错配供体选择，统一无额外影子推理。原400动作/60000物理步/1200秒上限及任务判据不变。此为技术验证，不用于确认触觉收益。

## 结果

状态：{result['status']}；官方成功：{result.get('success','未知')}；结束原因：{result.get('termination',result.get('error','未知'))}；资源截断：{result.get('resource_censored','未知')}；实际请求数：{len(requests)}；动作数：{result.get('actions','未知')}。

![执行频率、触觉年龄与接触记录](timing_contact.png)

{image_note}

## 失败分析与局限

错误：{result.get('error','无已记录异常；仍须完整配对审计')}。最后相机画面是请求前记录，不代表动作后终态。资源截断、推理失败、数据异常不作为普通任务失败计分；技术门必须无这些问题。推理墙钟耗时不等同于仿真物理时间。图中触觉未经物理校准，年龄不替代有效性掩码。两个开发seed不能估计可靠收益，不能进入正式确认或专家训练。

## 复现与下一步

协议CONTROLLED_FEEDBACK_PILOT_PROTOCOL_V1_ZH.md，冻结清单data/reports/controlled_feedback_pilot_v1/manifest.json。原始记录：{trial}。summary.json保留结果及来源SHA。完成固定16回合后核验执行频率、初态首命令、原始同源数据与资源预算，再决定是否具备正式确认条件。不补选成功seed，不自动启动Strata。
''')

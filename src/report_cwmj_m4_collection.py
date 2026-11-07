"""One complete illustrated report for each attempted M1 collection seed."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_cwmj_m4 import FOLDS,REPO
from validate_expert_data import read_jsonl,sha256

def report(seed):
    fold=FOLDS[seed];trial=REPO/f'data/paired/relay_pickcwmj_s{seed}_online_h50_stride20_{fold}_expert0'
    out=REPO/f'data/reports/cwmj_m4/seed_{seed}';out.mkdir(exist_ok=False)
    read=lambda n:json.loads((trial/n).read_text()) if (trial/n).exists() else None
    result=read('pilot_result.json');failure=read('failure.json');complete=read('completion.json')
    audit=None;samples=[];error=None
    if complete:
        try:
            from cwmj_m4_samples import M4TerminalEpisode
            ep=M4TerminalEpisode(seed);samples=[ep.sample(i,False) for i in range(len(ep))]
            audit=ep.reader.report
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    raws=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl'))]
    rows=read_jsonl(raws[0])[1:] if len(raws)==1 else []
    status='accepted_success_source_verified' if audit else ('source_rejected' if complete else 'collection_rejected')
    summary={'seed':seed,'fold':fold,'status':status,'pilot_result':result,'failure':failure,'audit_error':error,
             'valid_windows':len(samples),'contact_windows':sum(s['metadata']['contact'] for s in samples),
             'physics_steps':len(rows),'raw_contact_steps':sum(any(r[k]['contacts']>0 for k in ['left','right']) for r in rows),
             'source_audit':audit,'collector_manifest':json.loads((REPO/f'work/cwmj_m4_{seed}/source_manifest.json').read_text())}
    (out/'result.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(figsize=(9,3),layout='constrained')
    if rows:
        for side in ['left','right']:ax.plot([r['physics_step'] for r in rows],[r[side]['load_proxy_n'] for r in rows],label=side,lw=.8)
        ax.set_ylabel('Impulse / dt proxy (N)');ax.set_xlabel('Recorded physical step');ax.legend();ax.set_yscale('symlog',linthresh=.1)
    else:ax.text(.5,.5,'No recorded physics steps\n'+str((failure or {}).get('exception_type','preflight rejected')),ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
    ax.set_title(f'Seed {seed} | {fold} | {status}');fig.savefig(out/'contact_trace.png',dpi=160);plt.close(fig)
    image_text=''
    if audit:
        import h5py,cv2
        ep=__import__('cwmj_m4_samples').M4TerminalEpisode(seed)
        first_contact=next((s['metadata']['frame_index'] for s in samples if s['metadata']['contact']),len(ep.reader.frames)-1)
        ids=[0,first_contact,len(ep.reader.frames)-1]
        with h5py.File(ep.reader.hdf5) as h:
            fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
            for ax,i,label in zip(axes,ids,['Initial','First causal contact','Final captured pre-action']):
                img=cv2.imdecode(np.frombuffer(h['observation/head_camera/rgb'][i],np.uint8),cv2.IMREAD_COLOR)
                ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.set_title(f'{label} (frame {i})',fontsize=9);ax.axis('off')
            fig.savefig(out/'camera_frames.png',dpi=160);plt.close(fig)
        image_text='\n![同回合相机抽帧](camera_frames.png)\n'
    text=f'''# M4 CWMJ数据采集实验：seed {seed}

## 目的与预登记

扩充独立 {fold} 回合，检查当前观测、过去触觉及真实未来动作的同源性。seed 在采集前固定，不重试或按结果换 seed。CWMJ单酒瓶场景 / 原 pick_diverse_bottles 成功判据 / stride20 在线规则专家；128 action、10000 物理步、900 秒预算。用户持续授权本机 GPU 仿真；原源码/资产不修改；隔离进程实际替换为CWMJ酒瓶，scenario.json单独绑定工厂与资产SHA，与原20种瓶场景分开。

## 结果

| 项目 | 数值 |
|---|---|
| 状态 | {status} |
| 因果有效窗口 | {len(samples)} |
| 含过去接触窗口 | {summary['contact_windows']} |
| 记录物理步 | {len(rows)} |
| 有手指接触物理步 | {summary['raw_contact_steps']} |
| 数据异常 | {error or '无已检测到的源校验异常（未完成回合不进入校验）'} |

![左右手接触曲线](contact_trace.png)
{image_text}
## 方法、失败分析和解释边界

```json
{json.dumps({'pilot_result':result,'failure':failure},ensure_ascii=False,indent=2)}
```

仅完整成功且通过原 HDF5/物理时钟/控制器/源 SHA 审计的回合可进入本批候选训练或验证。未完成回合不提供专家标签；初始化异常不算策略失败。相机末帧发生在最后动作前，不能以末帧替代最终成功判据。接触量是仿真冲量/时间代理，不是实测传感器数据。

本实验不训练模型，也不证明触觉收益。当前规则专家不是失败恢复专家，失败记录只能用于后续恢复协议设计。下一步是汇总固定六个 seed 的有效数据量，达到至少两个新成功训练 seed 及一个新验证 seed 才开始本批训练。

## 复现

原始目录：`{trial}`。生成器及隔离采集源码 SHA、完整源数据 SHA（若通过）见相邻 `result.json`。批协议：`{REPO}/docs/CWMJ_M4_COLLECTION_PROTOCOL.md`。每个 GPU 任务退出释放自身租约，不自动启动 Strata。
'''
    (out/'report.md').write_text(text)
    dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m4')/f'seed_{seed}';dest.parent.mkdir(exist_ok=True);shutil.copytree(out,dest)
    print(json.dumps({k:summary[k] for k in ['seed','fold','status','valid_windows','contact_windows','physics_steps']}),flush=True)
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=FOLDS,required=True);report(p.parse_args().seed)

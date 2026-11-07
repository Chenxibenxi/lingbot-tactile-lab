"""Illustrated completed/failed recovery attempts without invented terminal data."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_expert_data import read_jsonl,sha256


def report(trial,out,*,publish=True):
    trial=Path(trial);out=Path(out)
    result_path=trial/'result.json';failure_path=trial/'failure.json'
    if not result_path.exists() and not failure_path.exists():raise ValueError('terminal or failure record required; do not report a live attempt')
    result=json.loads((result_path if result_path.exists() else failure_path).read_text())
    journal=trial/'action_journal.jsonl'
    events=read_jsonl(journal) if journal.exists() else []
    before=[x for x in events if x.get('event')=='before_action']
    after=[x for x in events if x.get('event')=='after_action']
    interrupted=sorted({x['action_index'] for x in before}-{x['action_index'] for x in after})
    out.mkdir(parents=True,exist_ok=False)
    fig,axs=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for side,index in [('left',6),('right',13)]:
        axs[0].plot([x['action_index'] for x in after],[x['command_state_14'][index] for x in after],label=side)
    for x in after:
        if x['source_role']=='external_intervention':axs[0].axvline(x['action_index'],color='purple',ls='--',label='Executed intervention')
    for i in interrupted:axs[0].axvline(i,color='grey',ls=':',label='Attempt without completion marker')
    axs[0].set(xlabel='Action index',ylabel='Gripper command',title='Executed commands and intervention');axs[0].legend(fontsize=7)
    raw=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl'))]
    if len(raw)==1:
        physics=read_jsonl(raw[0])[1:]
        for side in ('left','right'):
            axs[1].plot([x['physics_step'] for x in physics],[x[side]['load_proxy_n'] for x in physics],label=side,lw=.6)
        axs[1].set_yscale('symlog',linthresh=.1);axs[1].legend()
    else:axs[1].text(.5,.5,'No unique recorded physics stream',ha='center',transform=axs[1].transAxes)
    axs[1].set(xlabel='Physical step',ylabel='Uncalibrated impulse/dt proxy',title='Recorded contact')
    fig.savefig(out/'actions_contact.png',dpi=150);plt.close(fig)
    image_note='未取得可读取的HDF5相机帧；不补造图像。'
    hdf5=trial/'demonstration/data/episode0.hdf5'
    if hdf5.exists():
        import h5py,cv2
        try:
            with h5py.File(hdf5) as h:
                images=h['observation/head_camera/rgb'];n=len(images)
                if n:
                    intervention=next((x['camera_frame_index'] for x in after if x['source_role']=='external_intervention'),None)
                    chosen=[(0,'Initial pre-action')]
                    if intervention is not None and intervention<n:chosen.append((intervention,'Before intervention'))
                    chosen.append((n-1,'Last recorded pre-action'))
                    decoded=[]
                    for i,label in chosen:
                        img=cv2.imdecode(np.frombuffer(images[i],np.uint8),cv2.IMREAD_COLOR)
                        if img is None:raise ValueError('camera frame decode failed')
                        decoded.append((cv2.cvtColor(img,cv2.COLOR_BGR2RGB),i,label))
                    fig,axs=plt.subplots(1,len(decoded),figsize=(10,3),layout='constrained')
                    for ax,(img,i,label) in zip(np.atleast_1d(axs),decoded):ax.imshow(img);ax.set_title(f'{label} ({i})',fontsize=9);ax.axis('off')
                    fig.savefig(out/'camera_frames.png',dpi=150);plt.close(fig)
                    image_note='![相机记录](camera_frames.png)'
        except (OSError,KeyError,ValueError) as exc:image_note=f'相机产物不可读：{type(exc).__name__}: {exc}。未替换为其他回合画面。'
    evidence={str(p):sha256(p) for p in (result_path,failure_path,journal,hdf5) if p.is_file()}
    summary=dict(result=result,completed_action_markers=len(after),unresolved_action_attempts=interrupted,evidence_sha256=evidence)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'report.md').write_text(f'''# 恢复开发试验：seed {result['seed']} / {result['condition']}

## 方法与数据

固定四个开发seed、无扰动与一次左爪打开恢复配对。当前场景专家在lift后、place前接受一次开爪干预，再重新规划grasp/lift/place；控制条件不额外操作。原任务判据保持不变，最多400动作/60000物理步/1200秒。干预动作不是专家标签，H50标签不跨越干预。本报告不评估学习后的VLA。

## 结果

记录状态：{result['status']}；官方成功：{result.get('official_success','未知，未取得可评分终态')}；结束原因：{result.get('termination',result.get('error_type','未知'))}。已完成动作标记{len(after)}；仅有尝试、缺少完成标记的动作：{interrupted}。这些动作不被假定为执行成功或执行失败。

![动作与接触记录](actions_contact.png)

{image_note}

## 失败分析与局限

错误或缺失详情：{result.get('error','无已记录异常；仍需成对初态和数据来源审计')}。相机均为动作前采集，最后图片不是动作后终态；成功以原判据为准。触觉为未校准仿真代理量。单回合或四对开发样本不能证明稳定恢复率，更不能证明触觉模型收益。成功数据也不自动授权训练；外部扰动、阶段和几何真值不进入学习者输入。

## 复现与下一步

原始产物：`{trial}`。summary.json记录源文件SHA和原结果；action_journal.jsonl逐动作区分执行前后，未完成尝试保留未知状态。固定协议RECOVERY_DEVELOPMENT_PROTOCOL_V1_ZH.md；完成全部8回合后按预定可行性门汇总，不补选成功seed。原始仓库、checkpoint保持只读；不自动重启Strata。
''')
    if publish:
        dest=Path('/home/admin001/lingbot-tactile-artifacts/recovery-development')/trial.name
        dest.parent.mkdir(exist_ok=True);shutil.copytree(out,dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report(a.trial,a.out)

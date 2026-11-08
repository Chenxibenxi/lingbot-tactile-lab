"""Report v2 attempts including mismatches; never infer success from pictures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from validate_expert_data import read_jsonl, sha256


def report(trial, out):
    trial, out = Path(trial), Path(out)
    terminal = trial/'failure.json' if (trial/'failure.json').exists() else trial/'result.json'
    if not terminal.exists(): raise ValueError('terminal record required')
    result = json.loads(terminal.read_text())
    events = read_jsonl(trial/'action_journal.jsonl')
    completed = [x for x in events if x.get('event') == 'captured_controller_completed']
    attempted = [x for x in events if x.get('event') == 'captured_before_controller']
    unresolved = sorted({x['camera_frame_index'] for x in attempted}-{x['camera_frame_index'] for x in completed})
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for side,index in (('left',6),('right',13)):
        axes[0].plot([x['before_physics_step'] for x in completed],
                     [x['command_state_14'][index] for x in completed], label=side)
    for event in events:
        if event.get('event') == 'branch_after_action' and event.get('source_role') == 'external_intervention':
            axes[0].axvline(event['before_physics_step'],color='purple',ls='--')
    axes[0].set(title='Executed gripper commands',xlabel='Physical step',ylabel='Normalized command')
    axes[0].legend()
    if not completed:
        axes[0].text(.5,.5,'No completed action records',ha='center',transform=axes[0].transAxes)
    raw = [p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl'))]
    if len(raw) == 1:
        physics = read_jsonl(raw[0])[1:]
        for side in ('left','right'):
            axes[1].plot([x['physics_step'] for x in physics], [x[side]['load_proxy_n'] for x in physics],label=side,lw=.6)
        axes[1].set_yscale('symlog',linthresh=.1); axes[1].legend()
    else:
        axes[1].text(.5,.5,'No unique recorded physics stream',ha='center',transform=axes[1].transAxes)
    axes[1].set(title='Recorded contact',xlabel='Physical step',ylabel='Uncalibrated impulse/dt proxy')
    fig.savefig(out/'actions_contact.png',dpi=140); plt.close(fig)
    snapshots = sorted((trial/'snapshots').glob('*/snapshot.json'),key=lambda p:int(p.parent.name))
    terminal_obs = json.loads((trial/'terminal_observation.json').read_text()) if (trial/'terminal_observation.json').exists() else None
    if snapshots:
        fig, axes = plt.subplots(1,len(snapshots),figsize=(4*len(snapshots),3.6),squeeze=False,layout='constrained')
        for axis,path in zip(axes[0],snapshots):
            metadata = json.loads(path.read_text())
            is_terminal = terminal_obs is not None and int(path.parent.name) == terminal_obs['snapshot_index']
            label = 'Recorded terminal' if is_terminal else ('Initial' if path.parent.name == '0' else 'Prefix boundary')
            with Image.open(path.parent/'head_camera.png') as img: axis.imshow(img.copy())
            axis.set_title(f"{label}\nphysics step {metadata['physics_step']}",fontsize=9);axis.axis('off')
        fig.savefig(out/'snapshots.png',dpi=140);plt.close(fig)
    evidence_paths = [terminal,trial/'action_journal.jsonl']+snapshots
    for path in snapshots: evidence_paths.extend(path.parent.glob('*.png'))
    summary = dict(result=result,completed_actions=len(completed),unresolved_attempts=unresolved,
                   state_audits=[x for x in events if x.get('event') in ('initial_state_audit','boundary_state_audit')],
                   evidence_sha256={str(p):sha256(p) for p in evidence_paths},training_approval=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    images = '![初始、边界与终态记录](snapshots.png)' if snapshots else '未取得状态图像，不补造画面。'
    (out/'report.md').write_text(f'''# 恢复开发 v2：{result['identity']['seed']} / {result['kind']}

## 方法与数据

四个固定开发 seed，每个生成一次 grasp/lift 公共前缀，再分别重置、重放并核验初始和边界状态。对照从边界放置；扰动组开左爪一次后从当前场景重新抓取、抬升、放置。原任务判据不变，stride=20，分支含前缀最多400动作/60000物理步/1200秒。生成回合最多128动作，其完成不代表任务成功。

## 结果

记录状态：{result['status']}。官方任务成功：{result.get('official_success','未知')}。完成动作标记：{len(completed)}；已采集但无控制器完成标记：{unresolved}。前缀生成的 official_success=false 表示未评估完整任务，不应计入控制成功率。

![动作与触觉代理量](actions_contact.png)

{images}

## 失败分析与局限

记录错误：{result.get('error','无已记录异常；仍须完整配对汇总审计')}。有 terminal_observation 才标为动作后终态，其他图像只标初始或边界。图像不能替代官方判据；状态核验不覆盖 PhysX 隐藏缓存。触觉为未校准代理量。开发样本不能证明学习后模型收益，所有数据训练准入仍为 false。未知回合保留在固定分母。

## 复现与下一步

协议：RECOVERY_DEVELOPMENT_PROTOCOL_V2_ZH.md。原始产物：{trial}。源与协议哈希见 preregistration.json、result.json，图文来源哈希见 summary.json。按预定四对门槛汇总，不替换失败 seed；通过后另立正式采集与训练协议。原仓库和 checkpoint 只读，不自动重启 Strata。
''')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trial',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();report(args.trial,args.out)

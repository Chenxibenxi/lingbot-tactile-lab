"""Read terminal artifacts into an explicitly incomplete navigation index.

This does not select trials, compute interim hypothesis tests, or edit frozen
confirmation inputs. The final matrix report remains authoritative.
"""
from pathlib import Path
from datetime import datetime, timezone
import json

R = Path(__file__).resolve().parents[1]
DEST = Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m6-confirmation')
CONDITIONS = ('zero', 'true', 'mismatch', 'true_input_zero')


def report():
    rows = []
    pending_reports = []
    for seed in range(100071, 100087):
        for repeat in range(3):
            for arm in CONDITIONS:
                name = f'r{repeat}_{seed}_{arm}'
                path = R / f'data/eval/cwmj_m6_confirm_{name}/result.json'
                if not path.exists():
                    continue
                # A writer can still be closing a result file: do not infer a
                # completed outcome from an incomplete JSON snapshot.
                try:
                    result = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                ready = (DEST / name / 'report.md').exists()
                row = dict(name=name, seed=seed, repeat=repeat, condition=arm,
                           status=result.get('status'), success=result.get('success'),
                           termination=result.get('termination', result.get('exception')),
                           actions=result.get('actions'), report_ready=ready)
                rows.append(row)
                if not ready:
                    pending_reports.append(name)
    groups = []
    interruptions = []
    for path in sorted((R / 'data/reports').glob('cwmj_m6_confirmation_interruption_*.json')):
        record = json.loads(path.read_text())
        name = record['interrupted_attempt']
        if (R / f'data/eval/cwmj_m6_confirm_{name}/result.json').exists():
            raise ValueError(f'Interrupted attempt unexpectedly has a terminal result: {name}')
        interruptions.append(record)
    for path in sorted((R / 'data/reports/cwmj_m6_confirmation').glob('pair_r*.json')):
        groups.append(json.loads(path.read_text()))
    snapshot = dict(utc=datetime.now(timezone.utc).isoformat(), planned_episodes=192,
                    terminal_records=len(rows), report_count=sum(x['report_ready'] for x in rows),
                    pair_audits=len(groups), eligible_pairs=sum(x['eligible_complete_pair'] for x in groups),
                    scope='Navigation snapshot only; no interim efficacy inference or stopping rule',
                    interrupted_attempts=interruptions,
                    pending_reports=pending_reports, rows=rows)
    DEST.mkdir(exist_ok=True)
    (DEST / 'progress.json').write_text(json.dumps(snapshot, indent=2))
    lines = ['# M6独立仿真进度与报告索引', '',
             f"快照时间（UTC）：{snapshot['utc']}。已有终态 {len(rows)}/192，图文报告 {snapshot['report_count']}，四条件技术审计 {len(groups)} 组。", '',
             '这是执行进度快照，不是最终统计报告。未运行回合不算失败；已有回合也不能当作独立 seed 数。全部三次训练重复与四条件完成后，才按冻结协议汇总收益和不确定性。运行进程需单独核实，本索引不能证明队列仍活跃。', '',
             '| seed | 重复 | 条件/报告 | 状态 | 成功 | 动作 | 终止 |',
             '|---|---:|---|---|---|---:|---|']
    for row in rows:
        link = f"[{row['condition']}]({row['name']}/report.md)" if row['report_ready'] else row['condition'] + '（报告待生成）'
        lines.append(f"| {row['seed']} | {row['repeat']} | {link} | {row['status']} | {row['success']} | {row['actions']} | {row['termination']} |")
    if interruptions:
        lines += ['', '## 已确认中断（不计作任务失败）', '']
        for record in interruptions:
            name = record['interrupted_attempt']
            lines.append(f"- [{name}：中断审计]({name}/infrastructure_failure.md)。无完整终态，原始部分文件保留，不补跑。对应整 seed 从主要配对统计排除，其余预定条件继续执行。")
    (DEST / 'progress.md').write_text('\n'.join(lines) + '\n')
    compact = {k:v for k,v in snapshot.items() if k not in ('rows', 'pending_reports', 'interrupted_attempts')}
    compact['interrupted_attempt_count'] = len(interruptions)
    print(json.dumps(compact))


if __name__ == '__main__':
    report()

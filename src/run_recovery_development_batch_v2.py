"""Finite sequential batch: no backend lifecycle changes or GPU wait loop."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from record_recovery_development_v2 import R, KINDS, write_new
from report_recovery_development_v2 import report
from validate_expert_data import sha256


def verify_manifest():
    manifest = json.loads((R/'data/reports/recovery_development_v2/manifest.json').read_text())
    if manifest.get('gpu_execution_ready') is not True:
        raise ValueError('implementation not ready')
    for name,digest in manifest['files'].items():
        if sha256(R/name) != digest: raise ValueError('frozen source changed: '+name)
    return manifest


def resource_status():
    if shutil.disk_usage(R).free < 3*1024**3: return 'insufficient_disk'
    compute = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                             capture_output=True,text=True,check=True,timeout=10).stdout
    if compute.strip(): return 'gpu_busy'
    return 'available'


def run():
    manifest = verify_manifest()
    root = R/'data/reports/recovery_development_v2'
    # Exclusive invocation marker prevents a second autonomous launcher.
    write_new(root/'batch_started.json', dict(pid=os.getpid(),training_approval=False))
    rows = []
    for seed in manifest['seeds']:
        for kind in KINDS:
            verify_manifest()
            trial = R/f'data/paired/recovery_v2_{seed}_{kind}'
            if trial.exists(): raise ValueError('attempt already exists; inspect, never rerun')
            prefix = R/f'data/paired/recovery_v2_{seed}_prefix_generation/result.json'
            if kind != 'prefix_generation' and not prefix.exists():
                rows.append(dict(seed=seed,kind=kind,status='prerequisite_failed',official_success=None))
                continue
            state = resource_status()
            if state != 'available':
                write_new(root/'queue_result.json', dict(status='paused',reason=state,completed=rows,training_approval=False))
                return
            log_path = root/f'{seed}_{kind}.log'
            with log_path.open('x') as log:
                # The launcher owns timeout/kill and lease cleanup. Do not kill
                # only its parent shell with a competing Python timeout.
                result = subprocess.run(['bash',str(R/'scripts/run_recovery_development_v2.sh'),str(seed),kind],
                                        stdout=log,stderr=subprocess.STDOUT)
            terminal = trial/'failure.json' if (trial/'failure.json').exists() else trial/'result.json'
            if not terminal.exists():
                # Timeout/initialization failures must be explicit, never scored false.
                trial.mkdir(exist_ok=True)
                if not (trial/'action_journal.jsonl').exists(): (trial/'action_journal.jsonl').touch(exist_ok=False)
                write_new(trial/'failure.json',dict(status='needs_inspection',identity=dict(seed=seed),kind=kind,
                    error_type='LauncherFailure',error=f'launcher returncode={result.returncode}; no terminal record',training_approval=False))
                terminal = trial/'failure.json'
            record = json.loads(terminal.read_text())
            report(trial, root/f'reports/{seed}_{kind}')
            rows.append(dict(seed=seed,kind=kind,status=record['status'],
                             official_success=record.get('official_success'),returncode=result.returncode))
            write_new(root/f'{seed}_{kind}_receipt.json', rows[-1])
    write_new(root/'queue_result.json',dict(status='completed',attempts=rows,training_approval=False,
        limitation='Queue completion is not scientific gate approval; independent paired audit required.'))
    lines=['# 恢复开发 v2：固定批次状态','',
           '生成回合不计任务成功率；未知和前置失败保留分母。此表不替代配对审计或正式图文汇总。','',
           '| seed | 类型 | 状态 | 官方成功 |','|---|---|---|---|']
    lines += [f"| {x['seed']} | {x['kind']} | {x['status']} | {x['official_success']} |" for x in rows]
    (root/'batch_status.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': run()

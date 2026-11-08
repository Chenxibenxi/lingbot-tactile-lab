"""Conservative storage forecast from a complete technical pilot, no GPU work."""
import json
import math
import shutil
from pathlib import Path
from validate_expert_data import sha256


def forecast(rows):
    expected={(s,c,a) for s in (100145,100146) for c in (10,1) for a in ('zero','true','mismatch','contact_state')}
    actual=[(r['seed'],r['cadence'],r['arm']) for r in rows]
    if len(actual)!=16 or set(actual)!=expected:raise ValueError('full unique 16-trial pilot required')
    per_cadence={}
    for cadence in (10,1):
        group=[r for r in rows if r['cadence']==cadence]
        for r in group:
            if any(type(r[k]) is not int or r[k]<=0 for k in ('requests','bytes')) or r['requests']>400//cadence:
                raise ValueError('positive actual bytes/request counts required')
            if not math.isfinite(r['wall_seconds']) or r['wall_seconds']<=0:raise ValueError('valid runtime required')
        per_cadence[str(cadence)]=dict(
            max_bytes_per_request=max(r['bytes']/r['requests'] for r in group),
            max_runtime_seconds_per_request=max(r['wall_seconds']/r['requests'] for r in group),
            observed_total_bytes=sum(r['bytes'] for r in group),
            observed_max_requests=max(r['requests'] for r in group))
    # Includes fixed per-trial overhead in bytes/request, then extrapolates to full budget.
    full_bytes=sum(per_cadence[str(c)]['max_bytes_per_request']*(400//c)*192 for c in (10,1))
    policy_seconds=sum(per_cadence[str(c)]['max_runtime_seconds_per_request']*(400//c)*192 for c in (10,1))
    reserve=20*1024**3
    return dict(full_budget_bytes_estimate=math.ceil(full_bytes),
        required_initial_free_bytes=math.ceil(full_bytes*1.25)+reserve,
        keep_free_bytes=reserve,storage_headroom_factor=1.25,
        policy_runtime_seconds_estimate=policy_seconds,
        outer_timeout_total_seconds=384*1500,per_cadence=per_cadence,
        limitation='Worst observed per-request ratio extrapolated to 400 actions; image compression and scene complexity can change. Runtime excludes model loading and report generation; not a guaranteed completion time.')


def measure(root):
    root=Path(root);summary_path=root/'data/reports/controlled_feedback_pilot_v2_final/summary.json'
    summary=json.loads(summary_path.read_text())
    if summary.get('technical_gate') is not True:raise ValueError('complete independently audited pilot must pass first')
    rows=[]
    for episode in summary['episodes']:
        seed,cadence,arm=episode['seed'],episode['cadence'],episode['arm']
        p=root/f'data/eval/controlled_feedback_pilot_v2/s{seed}_{arm}_c{cadence}'
        if sha256(p/'result.json')!=episode['result_sha256']:raise ValueError('audited terminal result drift')
        result=json.loads((p/'result.json').read_text())
        paths=sorted(x for x in p.rglob('*') if x.is_file())
        rows.append(dict(seed=seed,cadence=cadence,arm=arm,requests=len(result['requests']),
            bytes=sum(x.stat().st_size for x in paths),wall_seconds=result['wall_seconds'],
            files=[dict(path=str(x.relative_to(p)),bytes=x.stat().st_size) for x in paths]))
    values=forecast(rows)
    volume=Path('/home/admin001/lingbot-tactile-storage')
    free=shutil.disk_usage(volume).free
    return dict(schema='feedback_confirmation_capacity_v1',pilot_summary_sha256=sha256(summary_path),
        forecast=values,observations=rows,output_volume=str(volume),actual_free_bytes=free,
        capacity_available=free>=values['required_initial_free_bytes'],
        execution_approved=False,auditor_sha256=sha256(Path(__file__)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=measure(Path(__file__).resolve().parents[1])
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='observations'}))

"""Persist per-episode analysis jobs; execute only with an available local service.

Run after the evaluation GPU batch, not concurrently. An unavailable service
leaves pending jobs without trying to load/restart it. Partial replies require
inspection and are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from strata_research_analysis import analyse,get

R=Path(__file__).resolve().parents[1]


def run(execute=False):
    root=R/'data/reports/cwmj_m8_strata';root.mkdir(exist_ok=True)
    jobs=[]
    for report in sorted((R/'data/reports/cwmj_m8_confirmation').glob('r*_s*_*_*/report.md')):
        paths=[report]+sorted(report.parent.glob('*.png'))
        job=dict(id=report.parent.name,report=str(report),images=[str(p) for p in paths[1:]],
                 evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        path=root/(job['id']+'.json')
        if path.exists():
            if json.loads(path.read_text())!=job:raise ValueError('queued evidence changed')
        else:
            with path.open('x') as f:json.dump(job,f,indent=2)
        jobs.append(job)
    if not execute:return dict(queued=len(jobs),status='queued_only')
    try:ready=get('/health').get('loaded') is True
    except (URLError,TimeoutError):ready=False
    if not ready:return dict(queued=len(jobs),status='waiting_for_already_loaded_strata')
    completed=0
    for job in jobs:
        out=root/job['id']
        if out.exists():
            if not (out/'analysis.md').exists():raise RuntimeError('partial analysis requires inspection: '+job['id'])
            request=json.loads((out/'request.json').read_text())
            if request['evidence_sha256']!=job['evidence_sha256']:raise ValueError('completed analysis input changed')
            completed+=1;continue
        analyse(job['report'],job['images'],out)
        completed+=1
    return dict(queued=len(jobs),completed=completed,status='completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    print(json.dumps(run(args.execute),ensure_ascii=False))

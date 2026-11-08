"""Frozen benchmark preflight; no GPU use and no process lifecycle changes."""
import json
from pathlib import Path
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]


def verify(root=R,require_feedback_finished=True):
    root=Path(root)
    manifest=root/'data/reports/microbatch_benchmark_v1/manifest.json'
    data=json.loads(manifest.read_text())
    if data.get('schema')!='microbatch_benchmark_frozen_v1' or data.get('frozen') is not True:
        raise ValueError('frozen benchmark manifest required')
    for name,digest in data['files'].items():
        if sha256(root/name)!=digest:raise ValueError('benchmark dependency drift: '+name)
    if require_feedback_finished:
        queue=json.loads((root/'data/reports/controlled_feedback_pilot_v1/queue_result.json').read_text())
        expected={(s,c,a) for s in (100127,100128) for c in (10,1) for a in ('zero','true','mismatch','contact_state')}
        attempts=queue.get('attempts',[])
        if queue.get('status')!='completed' or len(attempts)!=16 or {(x['seed'],x['cadence'],x['arm']) for x in attempts}!=expected:
            raise ValueError('feedback pilot is not terminal for all registered trials')
    return sha256(manifest)

if __name__=='__main__':print(verify())

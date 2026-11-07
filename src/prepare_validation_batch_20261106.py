"""Generate isolated SHA-pinned collectors for two preregistered validation seeds.
No GPU at generation. No retry or selection of a seed based on outcomes.
"""
from pathlib import Path
import argparse
import hashlib
import json
from patch_online_expert_stride20_pilot import REPO,OLD_PRE,OLD_REC,PRE_SHA,REC_SHA,_once

SEEDS=(100026,100027)

def generate(seed):
    if type(seed) is not int or seed not in SEEDS:raise ValueError('unregistered validation seed')
    pre=OLD_PRE.read_text();rec=OLD_REC.read_text()
    if hashlib.sha256(pre.encode()).hexdigest()!=PRE_SHA or hashlib.sha256(rec.encode()).hexdigest()!=REC_SHA:
        raise ValueError('frozen collector source drift')
    trial=f'relay_pickdiverse_s{seed}_online_h50_stride20_validation_expert0'
    pre=_once(pre,'REPO=Path(__file__).resolve().parents[1]',f'REPO=Path({str(REPO)!r})')
    pre=_once(pre,'SEED=100022',f'SEED={seed}')
    pre=_once(pre,"TRIAL='relay_pickdiverse_s100022_online_h50_expert0'",f'TRIAL={trial!r}')
    pre=_once(pre,"'fold_intent':'train_candidate_only_not_assigned'","'fold_intent':'validation_candidate_only_not_assigned'")
    rec=_once(rec,'from preregister_online_expert_trial import REPO,SEED,TASK,TRIAL,SOURCE,preregister',
        'from preregister_validation_batch import REPO,SEED,TASK,TRIAL,SOURCE,preregister')
    rec=_once(rec,"LEASE_OWNER='lingbot-tactile-lab/online-expert-s100022-v1'",
        f"LEASE_OWNER='lingbot-tactile-lab/validation-20261106-{seed}'")
    rec=_once(rec,'only one fresh relay s100022 online-rule expert trial',f'only one fresh relay s{seed} validation expert trial')
    rec=_once(rec,'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS)',
        'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS,stride=20)')
    if '100022' in rec or '100022' in pre:raise ValueError('old seed leaked')
    result={'preregister_validation_batch.py':pre,'record_validation_batch.py':rec}
    for name,text in result.items():compile(text,name,'exec')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    a=p.parse_args();out=REPO/f'work/validation_20261106_{a.seed}'
    files=generate(a.seed)
    if out.exists():raise FileExistsError('refuse collector replacement')
    out.mkdir()
    for name,text in files.items():
        with (out/name).open('x') as f:f.write(text)
    print(json.dumps({'seed':a.seed,'fold':'validation','files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}}))

if __name__=='__main__':main()

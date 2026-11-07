"""Generate isolated SHA-pinned collectors for sixteen fixed CWMJ M6 trials.
No GPU at generation. No retry or selection of a seed based on outcomes.
"""
from pathlib import Path
import argparse
import hashlib
import json
from patch_online_expert_stride20_pilot import REPO,OLD_PRE,OLD_REC,PRE_SHA,REC_SHA,_once

SEEDS=tuple(range(100055,100071))
FOLDS={s:('train' if s<100067 else 'validation') for s in SEEDS}

def generate(seed):
    if type(seed) is not int or seed not in SEEDS:raise ValueError('unregistered validation seed')
    fold=FOLDS[seed]
    pre=OLD_PRE.read_text();rec=OLD_REC.read_text()
    if hashlib.sha256(pre.encode()).hexdigest()!=PRE_SHA or hashlib.sha256(rec.encode()).hexdigest()!=REC_SHA:
        raise ValueError('frozen collector source drift')
    trial=f'relay_pickcwmjm6_s{seed}_online_h50_stride20_{fold}_expert0'
    pre=_once(pre,'REPO=Path(__file__).resolve().parents[1]',f'REPO=Path({str(REPO)!r})')
    pre=_once(pre,'SEED=100022',f'SEED={seed}')
    pre=_once(pre,"TRIAL='relay_pickdiverse_s100022_online_h50_expert0'",f'TRIAL={trial!r}')
    pre=_once(pre,"'fold_intent':'train_candidate_only_not_assigned'",f"'fold_intent':'{fold}_candidate_only_not_assigned'")
    rec=_once(rec,'from preregister_online_expert_trial import REPO,SEED,TASK,TRIAL,SOURCE,preregister',
        'from preregister_cwmj_m6 import REPO,SEED,TASK,TRIAL,SOURCE,preregister')
    rec=_once(rec,"LEASE_OWNER='lingbot-tactile-lab/online-expert-s100022-v1'",
        f"LEASE_OWNER='lingbot-tactile-lab/cwmj-m6-{seed}'")
    rec=_once(rec,'only one fresh relay s100022 online-rule expert trial',f'only one fresh relay s{seed} validation expert trial')
    rec=_once(rec,'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS)',
        'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS,stride=20)')
    rec=_once(rec,"    robot=getattr(importlib.import_module('envs.'+TASK),TASK)();closed=False;probe=None",
        "    from cwmj_bottle_factory import install\n    asset_evidence=install()\n    asset_evidence['factory_sha256']=sha256(REPO/'src/cwmj_bottle_factory.py')\n    (out/'scenario.json').write_text(json.dumps(asset_evidence,indent=2))\n    robot=getattr(importlib.import_module('envs.'+TASK),TASK)();closed=False;probe=None")
    if '100022' in rec or '100022' in pre:raise ValueError('old seed leaked')
    result={'preregister_cwmj_m6.py':pre,'record_cwmj_m6.py':rec}
    for name,text in result.items():compile(text,name,'exec')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    a=p.parse_args();out=REPO/f'work/cwmj_m6_{a.seed}'
    files=generate(a.seed)
    if out.exists():raise FileExistsError('refuse collector replacement')
    out.mkdir()
    for name,text in files.items():
        with (out/name).open('x') as f:f.write(text)
    print(json.dumps({'seed':a.seed,'fold':FOLDS[a.seed],'files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}}))

if __name__=='__main__':main()

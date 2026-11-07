"""Bounded stride10 data hypothesis: preserve 50 actual calls, no padding.

100028=train, 100029=validation. 128 calls/10000 physics steps/900s per trial.
More densely sampled planner path is a NEW collection protocol, not a measured
stride benefit. Generated files are isolated, exact hashes retained.
"""
import argparse
import hashlib
import json
from patch_online_expert_stride20_pilot import REPO,OLD_PRE,OLD_REC,PRE_SHA,REC_SHA,_once

SEED_FOLDS={100028:'train',100029:'validation'}

def generate(seed):
    if type(seed) is not int or seed not in SEED_FOLDS:raise ValueError('seed outside fixed batch')
    fold=SEED_FOLDS[seed]
    pre=OLD_PRE.read_text();rec=OLD_REC.read_text()
    if hashlib.sha256(pre.encode()).hexdigest()!=PRE_SHA or hashlib.sha256(rec.encode()).hexdigest()!=REC_SHA:
        raise ValueError('source SHA mismatch')
    trial=f'relay_pickdiverse_s{seed}_online_h50_stride10_{fold}_expert0'
    pre=_once(pre,'REPO=Path(__file__).resolve().parents[1]',f'REPO=Path({str(REPO)!r})')
    pre=_once(pre,'SEED=100022',f'SEED={seed}')
    pre=_once(pre,"TRIAL='relay_pickdiverse_s100022_online_h50_expert0'",f'TRIAL={trial!r}')
    pre=_once(pre,"'fold_intent':'train_candidate_only_not_assigned'",f"'fold_intent':'{fold}_candidate_only_not_assigned'")
    rec=_once(rec,'from preregister_online_expert_trial import REPO,SEED,TASK,TRIAL,SOURCE,preregister',
        'from preregister_contact_h50 import REPO,SEED,TASK,TRIAL,SOURCE,preregister')
    rec=_once(rec,'from online_expert_pilot_executor import execute_online_expert,MAX_PHYSICS_STEPS',
        'from contact_h50_executor import execute_online_expert,MAX_PHYSICS_STEPS')
    rec=_once(rec,"LEASE_OWNER='lingbot-tactile-lab/online-expert-s100022-v1'",f"LEASE_OWNER='lingbot-tactile-lab/contact-h50-{seed}'")
    rec=_once(rec,'only one fresh relay s100022 online-rule expert trial',f'only one fresh relay s{seed} contact H50 trial')
    rec=_once(rec,'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS)',
        'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS,stride=10)')
    proposal=(REPO/'src/online_bottle_expert_actions.py').read_text()
    proposal=_once(proposal,'EXPERIMENTAL_CPU_STRIDE=20','EXPERIMENTAL_CPU_STRIDE=10')
    executor=(REPO/'src/online_expert_pilot_executor.py').read_text()
    executor=_once(executor,'from online_bottle_expert_actions import','from contact_h50_proposals import')
    files={'preregister_contact_h50.py':pre,'record_contact_h50.py':rec,
           'contact_h50_executor.py':executor,'contact_h50_proposals.py':proposal}
    for name,text in files.items():compile(text,name,'exec')
    return files

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEED_FOLDS,required=True)
    a=p.parse_args();files=generate(a.seed);out=REPO/f'work/contact_h50_20261106_{a.seed}'
    out.mkdir(exist_ok=False)
    for name,text in files.items():
        with (out/name).open('x') as f:f.write(text)
    manifest={'seed':a.seed,'fold':SEED_FOLDS[a.seed],'stride':10,'max_calls':128,'max_physics_steps':10000,
              'files':{n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()},
              'source_files':{n:hashlib.sha256((REPO/'src'/n).read_bytes()).hexdigest() for n in
                              ('online_bottle_expert_actions.py','online_expert_pilot_executor.py')},'training_approval':False}
    with (out/'source_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(manifest))

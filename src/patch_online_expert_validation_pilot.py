"""SHA-pinned validation-ONLY private online collector from unchanged source.

Seed100024 is never a training fold or an excuse to run GPU until owner clears
workbench and exclusive one-off lease/budget preflight passes.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from patch_online_expert_stride20_pilot import REPO,OLD_PRE,OLD_REC,PRE_SHA,REC_SHA,_once

PRIVATE=REPO/'work/online_expert_stride20_s100024_validation_v1'
TRIAL='relay_pickdiverse_s100024_online_h50_stride20_validation_expert0'


def generate():
    pre=OLD_PRE.read_text();rec=OLD_REC.read_text()
    if (hashlib.sha256(pre.encode()).hexdigest()!=PRE_SHA or
            hashlib.sha256(rec.encode()).hexdigest()!=REC_SHA):
        raise ValueError('original frozen online collector SHA changed')
    pre=_once(pre,'REPO=Path(__file__).resolve().parents[1]',
                  'REPO=Path(__file__).resolve().parents[2]  # private validation work/ overlay')
    pre=_once(pre,'SEED=100022','SEED=100024')
    pre=_once(pre,"TRIAL='relay_pickdiverse_s100022_online_h50_expert0'",f"TRIAL='{TRIAL}'")
    pre=_once(pre,"'fold_intent':'train_candidate_only_not_assigned'",
                  "'fold_intent':'validation_candidate_only_not_assigned'")
    rec=_once(rec,'from preregister_online_expert_trial import REPO,SEED,TASK,TRIAL,SOURCE,preregister',
                  'from preregister_online_expert_validation import REPO,SEED,TASK,TRIAL,SOURCE,preregister')
    rec=_once(rec,"LEASE_OWNER='lingbot-tactile-lab/online-expert-s100022-v1'",
                  "LEASE_OWNER='lingbot-tactile-lab/online-expert-s100024-validation-v1'")
    rec=_once(rec,'only one fresh relay s100022 online-rule expert trial',
                  'only one fresh relay s100024 validation online-rule expert trial')
    rec=_once(rec,'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS)',
                  'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS,stride=20)')
    if '100022' in pre or '100022' in rec or '100023' in pre or '100023' in rec:
        raise ValueError('old seed/control leakage into new validation source')
    for name,text in [('preregister_online_expert_validation.py',pre),
                      ('record_online_expert_validation.py',rec)]:compile(text,name,'exec')
    return {'preregister_online_expert_validation.py':pre,
            'record_online_expert_validation.py':rec}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=PRIVATE)
    parser.add_argument('--create',action='store_true')
    args=parser.parse_args();out=args.out.resolve(strict=False)
    if out!=PRIVATE or args.create and out.exists():
        raise ValueError('only fresh private validation source directory')
    files=generate()
    if not args.create:
        if out.exists() and any(not (out/name).is_file() or (out/name).is_symlink() or
                                (out/name).read_text()!=text for name,text in files.items()):
            raise ValueError('private validation collector differs from pinned source')
        print('VALIDATION_CPU_DRY_RUN_NO_GPU_OR_NEW_DATA',TRIAL,
              {name:hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()})
        return
    out.mkdir(parents=True,exist_ok=False)
    for name,text in files.items():
        with (out/name).open('x',encoding='utf-8') as file:file.write(text)
    print('PRIVATE_VALIDATION_COLLECTOR_GENERATED_NOT_EXECUTED',TRIAL)


if __name__=='__main__':main()

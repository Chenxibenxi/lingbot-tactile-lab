"""Build one isolated validation-only seed100025 collector from pinned originals.

CPU preview only until the user manually stops GPU backends and the owner-only
lease/empty-3090 gate passes. This module does not change old seeds or Strata.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from patch_online_expert_stride20_pilot import REPO,OLD_PRE,OLD_REC,PRE_SHA,REC_SHA,_once

PRIVATE=REPO/'work/online_expert_stride20_s100025_validation_v1'
TRIAL='relay_pickdiverse_s100025_online_h50_stride20_validation_expert0'


def generate():
    pre=OLD_PRE.read_text();rec=OLD_REC.read_text()
    if (hashlib.sha256(pre.encode()).hexdigest()!=PRE_SHA or
            hashlib.sha256(rec.encode()).hexdigest()!=REC_SHA):
        raise ValueError('original online expert collector SHA changed')
    pre=_once(pre,'REPO=Path(__file__).resolve().parents[1]',
                  'REPO=Path(__file__).resolve().parents[2]  # one private validation overlay')
    pre=_once(pre,'SEED=100022','SEED=100025')
    pre=_once(pre,"TRIAL='relay_pickdiverse_s100022_online_h50_expert0'",f"TRIAL='{TRIAL}'")
    pre=_once(pre,"'fold_intent':'train_candidate_only_not_assigned'",
                  "'fold_intent':'validation_candidate_only_not_assigned'")
    rec=_once(rec,'from preregister_online_expert_trial import REPO,SEED,TASK,TRIAL,SOURCE,preregister',
                  'from preregister_online_expert_validation_s100025 import REPO,SEED,TASK,TRIAL,SOURCE,preregister')
    rec=_once(rec,"LEASE_OWNER='lingbot-tactile-lab/online-expert-s100022-v1'",
                  "LEASE_OWNER='lingbot-tactile-lab/online-expert-s100025-validation-v1'")
    rec=_once(rec,'only one fresh relay s100022 online-rule expert trial',
                  'only one fresh relay s100025 validation online-rule expert trial')
    rec=_once(rec,'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS)',
                  'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS,stride=20)')
    if any(str(seed) in pre or str(seed) in rec for seed in (100022,100023,100024)):
        raise ValueError('old seed leakage into independent validation collector')
    for name,text in [('preregister_online_expert_validation_s100025.py',pre),
                      ('record_online_expert_validation_s100025.py',rec)]:compile(text,name,'exec')
    return {'preregister_online_expert_validation_s100025.py':pre,
            'record_online_expert_validation_s100025.py':rec}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=PRIVATE)
    p.add_argument('--create',action='store_true')
    args=p.parse_args();out=args.out.resolve(strict=False)
    if out!=PRIVATE or (args.create and out.exists()):
        raise ValueError('only one fresh private validation preview directory')
    files=generate()
    if not args.create:
        if out.exists() and any(not (out/name).is_file() or (out/name).is_symlink() or
                                (out/name).read_text()!=text for name,text in files.items()):
            raise ValueError('private s100025 validation code differs from pinned source')
        print('VALIDATION_100025_CPU_DRY_RUN_NO_GPU',TRIAL,
              {name:hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()})
        return
    out.mkdir(parents=True,exist_ok=False)
    for name,text in files.items():
        with (out/name).open('x',encoding='utf-8') as f:f.write(text)
    print('VALIDATION_100025_PRIVATE_PREVIEW_CREATED_NOT_EXECUTED',TRIAL)


if __name__=='__main__':main()

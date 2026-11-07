"""SHA-locked isolated copy of the old safe online collector for ONE new seed.

CPU --dry-run emits no files/GPU work; --create only writes private work/ files.
Original s100022 collector/preregistration and all old private trials are untouched.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
OLD_PRE=REPO/'src/preregister_online_expert_trial.py'
OLD_REC=REPO/'src/record_online_expert_demo.py'
PRE_SHA='a19e1614cebe545a553a00d8e3cdc1a6ab87fba705b6723ea3861780bb47ae79'
REC_SHA='fd7f956ee3f92ccb949ff59a8ee27f6cc875decf12eba29869fb33a5f14c837c'
PRIVATE=REPO/'work/online_expert_stride20_s100023_v1'
TRIAL='relay_pickdiverse_s100023_online_h50_stride20_expert0'


def _once(text,old,new):
    if text.count(old)!=1:raise ValueError(f'unsupported original online expert source anchor {old!r}')
    return text.replace(old,new)


def generate():
    pre=OLD_PRE.read_text();rec=OLD_REC.read_text()
    if (hashlib.sha256(pre.encode()).hexdigest()!=PRE_SHA or
            hashlib.sha256(rec.encode()).hexdigest()!=REC_SHA):
        raise ValueError('original seed100022 preregistration/collector source SHA changed')
    pre=_once(pre,"REPO=Path(__file__).resolve().parents[1]",
                  "REPO=Path(__file__).resolve().parents[2]  # private work/ overlay, original repo")
    pre=_once(pre,'SEED=100022','SEED=100023')
    pre=_once(pre,"TRIAL='relay_pickdiverse_s100022_online_h50_expert0'",f"TRIAL='{TRIAL}'")
    rec=_once(rec,'from preregister_online_expert_trial import REPO,SEED,TASK,TRIAL,SOURCE,preregister',
                  'from preregister_online_expert_stride20 import REPO,SEED,TASK,TRIAL,SOURCE,preregister')
    rec=_once(rec,"LEASE_OWNER='lingbot-tactile-lab/online-expert-s100022-v1'",
                  "LEASE_OWNER='lingbot-tactile-lab/online-expert-s100023-stride20-v1'")
    rec=_once(rec,"only one fresh relay s100022 online-rule expert trial",
                  "only one fresh relay s100023 stride20 online-rule expert trial")
    rec=_once(rec,'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS)',
                  'execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS,stride=20)')
    if '100022' in pre or '100022' in rec:raise ValueError('old seed leakage into new isolated pilot')
    for name,text in [('preregister_online_expert_stride20.py',pre),
                      ('record_online_expert_stride20.py',rec)]:
        compile(text,name,'exec')
    return {'preregister_online_expert_stride20.py':pre,
            'record_online_expert_stride20.py':rec}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=PRIVATE)
    parser.add_argument('--create',action='store_true')
    args=parser.parse_args()
    out=args.out.resolve(strict=False)
    if out!=PRIVATE or (args.create and out.exists()):
        raise ValueError('only fresh SHA-locked private work/ directory')
    generated=generate()
    if not args.create:
        if out.exists() and any(not (out/name).is_file() or (out/name).is_symlink() or
                                (out/name).read_text()!=text for name,text in generated.items()):
            raise ValueError('private generated new-seed collector differs from pinned source')
        print('CPU_DRY_RUN_ONLY_NO_GPU_OR_NEW_SEED',TRIAL,
              {name:hashlib.sha256(text.encode()).hexdigest() for name,text in generated.items()})
        return
    out.mkdir(parents=True,exist_ok=False)
    for name,text in generated.items():
        with (out/name).open('x',encoding='utf-8') as file:file.write(text)
    print('PRIVATE_NEWSEED_COLLECTOR_GENERATED_NOT_EXECUTED',TRIAL)


if __name__=='__main__':main()

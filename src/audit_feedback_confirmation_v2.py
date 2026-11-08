"""Full fixed-matrix audit. Unknown trials cannot become ordinary failures."""
import json
from pathlib import Path
import numpy as np
from validate_expert_data import sha256
from validate_policy_rollout import validate
from feedback_outcome_contract_v2 import validate_outcome
from audit_controlled_feedback_timing import audit_hdf5
from audit_feedback_pilot_v2_final import compare_initial
from verify_feedback_confirmation_v2 import SEEDS,ARMS,identity
from feedback_confirmation_statistics import summarize


def analyze_complete(rows,pair_checks):
    expected={(s,r,c,a) for s in SEEDS for r in (0,1,2) for c in (10,1) for a in ARMS}
    actual=[(x['seed'],x['repeat'],x['cadence'],x['arm']) for x in rows]
    if len(actual)!=384 or set(actual)!=expected:raise ValueError('complete 384-attempt matrix required')
    expected_pairs={f'{s}_r{r}' for s in SEEDS for r in (0,1,2)}
    if set(pair_checks)!=expected_pairs:raise ValueError('all 48 within-repeat pairing checks required')
    valid=all(x['technical_pass'] is True and type(x['success']) is bool for x in rows) and all(v is True for v in pair_checks.values())
    if not valid:return dict(confirmation_gate=False,statistics=None,reason='incomplete technical or pairing evidence; unknown is not zero')
    x=np.zeros((16,3,2,4),dtype=np.int64)
    for row in rows:x[SEEDS.index(row['seed']),row['repeat'],(10,1).index(row['cadence']),ARMS.index(row['arm'])]=int(row['success'])
    return dict(confirmation_gate=True,statistics=summarize(x,SEEDS),reason=None)


def audit(root):
    root=Path(root);base=root/'data/reports/feedback_confirmation_v2'
    queue=json.loads((base/'queue_result.json').read_text())
    expected={(s,r,c,a) for s in SEEDS for r in (0,1,2) for c in (10,1) for a in ARMS}
    actual=[(x['seed'],x['repeat'],x['cadence'],x['arm']) for x in queue.get('attempts',[])]
    if queue.get('status')!='completed' or len(actual)!=384 or set(actual)!=expected:
        raise ValueError('all registered attempts must be terminal')
    manifest=json.loads((base/'manifest.json').read_text());digest=sha256(base/'manifest.json')
    for name,value in manifest['files'].items():
        if sha256(root/name)!=value:raise ValueError('frozen source drift: '+name)
    rows=[];eligible={}
    for seed,repeat,cadence,arm in sorted(expected):
        trial=root/'data/eval/feedback_confirmation_v2'/identity(seed,arm,repeat,cadence)
        row=dict(seed=seed,repeat=repeat,cadence=cadence,arm=arm,success=None,technical_pass=False)
        try:
            result=json.loads((trial/'result.json').read_text());row['result_sha256']=sha256(trial/'result.json')
            row['status']=result['status'];row['reported_success']=result.get('success')
            if result['status']!='completed' or result.get('resource_censored') is not False:
                raise ValueError(result.get('error',result.get('termination','missing resource status')))
            validate_outcome(result)
            if (result['seed'],result['repeat_index'],result['replan_steps'],result['arm'])!=(seed,repeat,cadence,arm):
                raise ValueError('result identity mismatch')
            if result['training_steps']!=2880 or result['confirmation_manifest_sha256']!=digest:
                raise ValueError('checkpoint or manifest mismatch')
            if type(result['success']) is not bool or result['termination'] not in ('task_success','official_action_limit') or result['success']!=(result['termination']=='task_success'):
                raise ValueError('invalid official terminal result')
            if result['adapter_sha256']!=manifest['adapters'][f'r{repeat}_{arm}']['sha256']:
                raise ValueError('wrong adapter')
            raws=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.policy.jsonl','.frames.jsonl'))]
            if len(raws)!=1:raise ValueError('unique raw physics source required')
            raw=raws[0]
            source=validate(raw,raw.with_suffix('.obs.jsonl'),raw.with_suffix('.policy.jsonl'),trial/'rollout.hdf5')
            if source!=result['source_audit'] or source['seed']!=seed or source['executed_action_count']!=result['actions']:
                raise ValueError('same-run source mismatch')
            timing=audit_hdf5(trial/'rollout.hdf5',result['requests'],seed=seed,cadence=cadence,official_success=result['success'])
            if timing!=result['cadence_audit']:raise ValueError('feedback cadence mismatch')
            from audit_feedback_topp import audit as audit_topp
            row['topp_audit']=audit_topp(json.loads((trial/'topp_events.json').read_text()),result['requests'],result['actions'])
            row['topp_sha256']=sha256(trial/'topp_events.json')
            row.update(success=result['success'],technical_pass=True,source_audit=source,cadence_audit=timing)
            eligible[(seed,repeat,cadence,arm)]=trial
        except (ValueError,KeyError,TypeError,OSError,AssertionError) as exc:
            row['error']=type(exc).__name__+': '+str(exc)
        rows.append(row)
    pairs={};pair_errors={}
    for seed in SEEDS:
        for repeat in (0,1,2):
            key=f'{seed}_r{repeat}';paths=[eligible.get((seed,repeat,c,a)) for c in (10,1) for a in ARMS]
            try:pairs[key]=all(p is not None for p in paths) and compare_initial(paths)
            except (ValueError,KeyError,TypeError,OSError,AssertionError) as exc:
                pairs[key]=False;pair_errors[key]=str(exc)
    analysis=analyze_complete(rows,pairs)
    return dict(schema='feedback_confirmation_audit_v1',terminal_contract_sha256=sha256(Path(__file__).with_name('feedback_outcome_contract_v2.py')),episodes=rows,pair_checks=pairs,pair_errors=pair_errors,
        fixed_denominator=384,unknown_trials=sum(x['success'] is None for x in rows),
        technical_gate=analysis['confirmation_gate'],statistics=analysis['statistics'],analysis_reason=analysis['reason'],
        training_approval=False,manifest_sha256=digest,auditor_sha256=sha256(Path(__file__)))

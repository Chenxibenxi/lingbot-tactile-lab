"""Post-run independent checks; never relax pairing after seeing outcomes."""
import json
from pathlib import Path
import h5py
import numpy as np
from audit_controlled_feedback_timing import audit_hdf5
from validate_policy_rollout import validate
from feedback_outcome_contract_v2 import validate_outcome
from validate_expert_data import sha256

ARMS=('zero','true','mismatch','contact_state')


def compare_initial(trials):
    if len(trials)!=8 or len(set(map(str,trials)))!=8:
        raise ValueError('eight distinct frequency/condition trials required')
    reference=None;scenarios=[]
    for trial in trials:
        with h5py.File(Path(trial)/'rollout.hdf5','r') as h:
            keys=('images/head','images/left','images/right','state_14','applied/command_14')
            current={key:h[key][0] for key in keys}
            if bool(h['tactile/window_valid'][0,0]):raise ValueError('initial tactile must be invalid')
            if reference is None:reference=current
            elif any(not np.array_equal(reference[key],current[key]) for key in keys):
                return False
        scenarios.append(json.loads((Path(trial)/'scenario.json').read_text()))
    return all(s==scenarios[0] for s in scenarios)


def audit(root):
    root=Path(root);base=root/'data/reports/controlled_feedback_pilot_v2'
    queue=json.loads((base/'queue_result.json').read_text())
    expected={(s,c,a) for s in (100145,100146) for c in (10,1) for a in ARMS}
    actual=[(x['seed'],x['cadence'],x['arm']) for x in queue.get('attempts',[])]
    if queue.get('status')!='completed' or len(actual)!=16 or set(actual)!=expected:
        raise ValueError('all sixteen terminal attempts required')
    manifest=json.loads((base/'manifest.json').read_text());manifest_sha=sha256(base/'manifest.json')
    for name,digest in manifest['files'].items():
        if sha256(root/name)!=digest:raise ValueError('frozen implementation drift')
    rows=[];pair_checks={};eligible={}
    for seed,cadence,arm in sorted(expected):
        trial=root/f'data/eval/controlled_feedback_pilot_v2/s{seed}_{arm}_c{cadence}'
        result=json.loads((trial/'result.json').read_text())
        row=dict(seed=seed,cadence=cadence,arm=arm,result_sha256=sha256(trial/'result.json'),
                 reported_success=result.get('success'),success=None,status=result['status'],technical_pass=False)
        if result['status']!='completed' or result.get('resource_censored') is not False:
            row['reason']=result.get('error',result.get('termination','missing resource status'))
            rows.append(row);continue
        validate_outcome(result)
        if (result['seed']!=seed or result['arm']!=arm or result['replan_steps']!=cadence
                or result['repeat_index']!=0 or result['training_steps']!=2880
                or result['pilot_manifest_sha256']!=manifest_sha or type(result['success']) is not bool
                or result['termination'] not in ('task_success','official_action_limit')):
            raise ValueError('result identity/terminal mismatch')
        if result['success']!=(result['termination']=='task_success'):raise ValueError('inconsistent outcome')
        trained=json.loads((root/f'data/reports/cwmj_m8_training_r0_{arm}/step_2880/result.json').read_text())
        if result['adapter_sha256']!=trained['sha256'] or manifest['artifacts'][trained['artifact']]!=trained['sha256']:
            raise ValueError('wrong frozen adapter')
        raw=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.policy.jsonl','.frames.jsonl'))]
        if len(raw)!=1:raise ValueError('unique raw physics required')
        raw=raw[0]
        source=validate(raw,raw.with_suffix('.obs.jsonl'),raw.with_suffix('.policy.jsonl'),trial/'rollout.hdf5')
        if source!=result['source_audit'] or source['seed']!=seed or source['executed_action_count']!=result['actions']:
            raise ValueError('same-run source audit mismatch')
        timing=audit_hdf5(trial/'rollout.hdf5',result['requests'],seed=seed,cadence=cadence,official_success=result['success'])
        if timing!=result['cadence_audit']:raise ValueError('actual cadence audit mismatch')
        from audit_feedback_topp import audit as audit_topp
        topp=audit_topp(json.loads((trial/'topp_events.json').read_text()),result['requests'],result['actions'])
        row['topp_audit']=topp
        row['topp_sha256']=sha256(trial/'topp_events.json')
        row.update(success=result['success'],technical_pass=True,source_audit=source,cadence_audit=timing)
        rows.append(row);eligible[(seed,cadence,arm)]=trial
    for seed in (100145,100146):
        paths=[eligible.get((seed,c,a)) for c in (10,1) for a in ARMS]
        pair_checks[str(seed)]=all(p is not None for p in paths) and compare_initial(paths)
    return dict(terminal_contract_sha256=sha256(Path(__file__).with_name('feedback_outcome_contract_v2.py')),episodes=rows,initial_first_command_pair_checks=pair_checks,
                technical_gate=all(x['technical_pass'] for x in rows) and all(pair_checks.values()),
                fixed_denominator=16,confirmation_eligible=False,training_approval=False,
                limitation='Technical feasibility only; success rates are descriptive, not confirmation evidence.')

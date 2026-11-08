"""Read-only post-run audit, separate from the frozen execution implementation."""
import json
from pathlib import Path
from validate_expert_data import sha256, read_jsonl
from recovery_pair_state_v2 import compare_states
from recovery_shared_prefix_v2 import load_prefix, validate_references
from recovery_source_audit_v2 import audit_sources

KINDS=('prefix_generation','expert_control','left_open_once_recovery')


def audit_seed(root,seed):
    root=Path(root)
    results={};errors={};evidence={}
    for kind in KINDS:
        trial=root/f'data/paired/recovery_v2_{seed}_{kind}'
        if (trial/'failure.json').exists():
            errors[kind]=json.loads((trial/'failure.json').read_text());continue
        if not (trial/'result.json').exists():
            errors[kind]={'status':'unavailable','reason':'no completed result'};continue
        result=json.loads((trial/'result.json').read_text())
        if result.get('status')!='completed' or result.get('kind')!=kind or result['identity']['seed']!=seed:
            raise ValueError('wrong completed identity')
        if result.get('training_approval') is not False:raise ValueError('unexpected training approval')
        if sha256(trial/'action_journal.jsonl')!=result['action_journal_sha256']:
            raise ValueError('journal changed')
        execution=json.loads((trial/'execution.json').read_text())
        paths={key:trial/rel for key,rel in result['source_paths'].items()}
        audit=audit_sources(**paths,actions=execution['rows'],seed=seed,physics_steps=execution['physics_steps'],
                            official_success=execution['official_success'],kind=kind)
        if audit!=result['source_audit']:raise ValueError('source audit changed')
        if result['official_success']!=execution['official_success']:raise ValueError('outcome changed')
        results[kind]=(result,execution)
        evidence[kind]={'result_sha256':sha256(trial/'result.json'),'execution_sha256':sha256(trial/'execution.json')}
    checks={}
    if 'prefix_generation' in results:
        pref=results['prefix_generation'][0]
        path=root/f'data/paired/recovery_v2_{seed}_prefix_generation/prefix.json'
        prefix=load_prefix(path,expected_sha256=pref['prefix_seal']['sha256'],expected_identity=pref['identity'])
        validate_references(prefix)
        for kind in KINDS[1:]:
            if kind not in results:continue
            result,execution=results[kind]
            if result['identity']!=pref['identity'] or result['prefix_seal']!=pref['prefix_seal']:
                raise ValueError('different sealed prefix identity')
            rows=execution['rows'];n=len(prefix['rows'])
            if len(rows)<n:raise ValueError('incomplete prefix execution')
            for actual,reference in zip(rows[:n],prefix['rows']):
                for key in ('action_index','phase','source_role','command_state_14','before_physics_step','after_physics_step'):
                    if actual[key]!=reference[key]:raise ValueError('prefix execution differs: '+key)
            journal=read_jsonl(root/f'data/paired/recovery_v2_{seed}_{kind}/action_journal.jsonl')
            for phase in ('initial','boundary'):
                matches=[x for x in journal if x.get('event')==phase+'_state_audit']
                if len(matches)!=1:raise ValueError('missing unique state audit')
                event=matches[0]
                if not compare_states(prefix[phase+'_state'],event['observed_state'])['observable_state_equal']:
                    raise ValueError('observable state mismatch')
                if event['observed_rgb_sha256']!=prefix[phase+'_rgb_sha256']:raise ValueError('RGB mismatch')
            intervention=[x for x in rows if x['source_role']=='external_intervention']
            if kind=='expert_control' and intervention:raise ValueError('control has intervention')
            if kind=='left_open_once_recovery' and (len(intervention)!=1 or intervention[0]['action_index']!=n):
                raise ValueError('intervention misplaced')
            checks[kind]=True
    return dict(seed=seed,technical_complete_pair=len(results)==3 and len(checks)==2 and not errors,
        outcomes={kind:results[kind][0]['official_success'] if kind in results else None for kind in KINDS[1:]},
        errors=errors,checks=checks,evidence=evidence,training_approval=False)

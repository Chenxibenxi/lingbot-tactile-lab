"""Freeze formal confirmation only after a complete independently audited pilot."""
import json
import subprocess
import sys
from pathlib import Path
from validate_expert_data import sha256
from verify_feedback_confirmation_v2 import R,SEEDS,ARMS
from audit_feedback_pilot_v2_final import audit
from feedback_capacity_budget_v2 import measure
from run_feedback_confirmation_batch_v2 import write


def consumed_seed_check():
    roots=[R/'data',Path('/home/admin001/lingbot-tactile-storage')]
    checked={};hits=[]
    for base in roots:
        for pattern in ('result.json','setup.json'):
            for p in base.rglob(pattern):
                path=p.resolve()
                if str(path) in checked:continue
                value=json.loads(path.read_text());checked[str(path)]=sha256(path)
                if value.get('seed') in SEEDS:hits.append(str(path))
    if hits:raise ValueError('candidate already has trial metadata: '+str(hits))
    for s in SEEDS:
        if any(str(s) in str(p) for base in roots for p in base.rglob(f'*{s}*')):
            raise ValueError('candidate trial path already exists')
    return dict(schema='feedback_formal_seed_admission_scan_v1',seeds=SEEDS,
        checked_trial_metadata=checked,conflicts=hits,
        limitation='Local result/setup records and seed-bearing paths; prior broad JSON provenance audit is retained separately.')


def prepare():
    root=R/'data/reports/feedback_confirmation_v2'
    formal=R/'docs/CONTROLLED_FEEDBACK_CONFIRMATION_PROTOCOL_V2_ZH.md'
    if root.exists() or formal.exists():raise FileExistsError('formal freeze already started; inspect rather than overwrite')
    prior=R/'data/reports/controlled_feedback_pilot_v2_final/summary.json'
    recorded=json.loads(prior.read_text());fresh=audit(R)
    if recorded.get('technical_gate') is not True or fresh['technical_gate'] is not True:
        raise ValueError('technical pilot must pass independently before formal freeze')
    for key in fresh:
        if fresh[key]!=recorded.get(key):raise ValueError('pilot audit changed: '+key)
    capacity=measure(R)
    if capacity['capacity_available'] is not True:raise ValueError('insufficient full-batch storage')
    seed_scan=consumed_seed_check()
    prior_seed_audit=R/'data/reports/feedback_candidate_seed_json_audit_v1.json'
    seed_history=json.loads(prior_seed_audit.read_text())
    if seed_history['candidate_seeds']!=SEEDS or seed_history.get('json_scope_clear') is not True:
        raise ValueError('prior JSON provenance audit not clear')
    from verify_cwmj_m8_confirmation import verify_confirmation
    files={};adapters={}
    for repeat in (0,1,2):
        for arm in ARMS:
            old=verify_confirmation(R,100103,repeat,arm,2880)
            for group in ('code_sha256','evidence_sha256'):
                for name,digest in old[group].items():
                    if sha256(R/name)!=digest:raise ValueError('historical dependency drift: '+name)
                    files[name]=digest
            node=R/f'data/reports/cwmj_m8_training_r{repeat}_{arm}/step_2880/result.json'
            meta=json.loads(node.read_text());weight=old['weights'][str(repeat)][arm]['2880']
            if any(meta[k]!=weight[k] for k in ('artifact','sha256')):raise ValueError('weight selection drift')
            adapters[f'r{repeat}_{arm}']=dict(artifact=meta['artifact'],sha256=meta['sha256'],result_sha256=sha256(node))
            files[str(node.relative_to(R))]=sha256(node)
    for name,digest in json.loads((R/'data/reports/controlled_feedback_pilot_v2/manifest.json').read_text())['files'].items():
        if sha256(R/name)!=digest:raise ValueError('pilot dependency drift: '+name)
        files[name]=digest
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(R/'tests'),'-p','test_feedback*.py'],check=True)
    subprocess.run(['bash','-n',str(R/'scripts/run_feedback_confirmation_v2.sh')],check=True)
    physical=Path('/home/admin001/lingbot-tactile-storage/feedback-confirmation-v2')
    logical=R/'data/eval/feedback_confirmation_v2'
    if physical.exists() or logical.exists() or logical.is_symlink():raise FileExistsError('fresh output directory required')
    # All read-only gates pass before creating the formal admission artifacts.
    physical.mkdir();logical.symlink_to(physical,target_is_directory=True);root.mkdir()
    write(root/'capacity.json',capacity);write(root/'seed_admission.json',seed_scan)
    template=R/'docs/CONTROLLED_FEEDBACK_CONFIRMATION_PROTOCOL_TEMPLATE_V2_ZH.md'
    with formal.open('x') as f:f.write(template.read_text().replace('协议模板（不是执行许可）','冻结执行协议',1))
    names=set(files)
    for pattern in ('*feedback_confirmation*.py','feedback_outcome_contract_v2.py','feedback_capacity_budget_v2.py','audit_feedback_pilot_v2_final.py'):
        names.update(str(p.relative_to(R)) for p in (R/'src').glob(pattern))
    names.update(str(p.relative_to(R)) for p in (R/'tests').glob('test_feedback*.py'))
    names.update(['scripts/run_feedback_confirmation_v2.sh',str(formal.relative_to(R)),str(template.relative_to(R)),
        str(prior.relative_to(R)),str(prior_seed_audit.relative_to(R)),
        str((root/'capacity.json').relative_to(R)),str((root/'seed_admission.json').relative_to(R))])
    manifest=dict(schema='feedback_confirmation_frozen_v2',gpu_execution_ready=True,seeds=SEEDS,
        repeats=[0,1,2],training_steps=2880,arms=list(ARMS),cadences=[10,1],adapters=adapters,
        minimum_free_output_bytes=capacity['forecast']['keep_free_bytes'],
        files={name:sha256(R/name) for name in sorted(names)},training_approval=False)
    write(root/'manifest.json',manifest)
    from verify_feedback_confirmation_v2 import verify
    verify(SEEDS[0],ARMS[0],0,10)
    print('Formal freeze ready:',sha256(root/'manifest.json'))

if __name__=='__main__':prepare()

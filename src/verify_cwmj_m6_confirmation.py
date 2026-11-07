"""Require complete fixed training evidence before consuming a confirmation seed."""
import json
from pathlib import Path
from validate_expert_data import sha256
from confirmation_tactile_controls import CONDITIONS,WEIGHT_ARM
CODE_FILES=['src/run_cwmj_m6_confirmation.py','src/confirmation_tactile_controls.py','src/verify_cwmj_m6_confirmation.py','src/probe_h50_training_backward.py','src/restore_nonpersistent_buffers.py','src/cwmj_bottle_factory.py','src/paired_seed_statistics.py','src/run_tactile_effect_pilot.py','src/tactile_adapter.py','src/tactile_rpc.py','src/window_math.py','src/force_probe.py','src/live_tactile_window.py','src/policy_rollout_capture.py','src/validate_policy_rollout.py','src/terminal_h50_contract.py','src/report_cwmj_m6_confirmation.py','src/report_cwmj_m6_group.py','src/audit_cwmj_m6_pair.py','src/report_cwmj_m6_confirmation_batch.py','work/run_cwmj_m6_confirmation.sh','work/run_cwmj_m6_confirmation_batch.py']
def verify_confirmation(repo,seed,repeat,condition):
    if type(seed) is not int or seed not in range(100071,100087) or repeat not in (0,1,2) or condition not in CONDITIONS:raise ValueError('fixed confirmation matrix only')
    root=repo/'data/reports/cwmj_m6';m=json.loads((root/'confirmation_manifest.json').read_text())
    if m['seeds']!=list(range(100071,100087)) or m['repetitions']!=[0,1,2] or m['conditions']!=list(CONDITIONS):raise ValueError('confirmation matrix drift')
    for name,key in [('docs/CWMJ_M6_TRAINING_PROTOCOL_ZH.md','protocol_sha256'),('data/reports/cwmj_m6/training_manifest.json','training_manifest_sha256'),('data/reports/cwmj_m6_training_summary.json','training_summary_sha256'),('data/reports/cwmj_m6/confirmation_donors.json','donor_manifest_sha256')]:
        if sha256(repo/name)!=m[key]:raise ValueError('confirmation evidence drift: '+name)
    for name,digest in m['code_sha256'].items():
        if sha256(repo/name)!=digest:raise ValueError('confirmation source drift: '+name)
    if set(m['code_sha256'])!=set(CODE_FILES):raise ValueError('missing pinned confirmation code')
    w=m['weights'][str(repeat)][WEIGHT_ARM[condition]]
    if sha256(Path(w['artifact']))!=w['sha256']:raise ValueError('fixed final-step checkpoint changed')
    if any((repo/'data/paired').glob(f'*s{seed}*')):raise ValueError('confirmation seed used in expert collection')
    for p in (repo/'data/eval').glob(f'*{seed}*'):
        if not p.name.startswith('cwmj_m6_confirm_'):raise ValueError('confirmation seed has earlier evaluation')
    return m

"""Candidate analysis contract; not permission to run or promote pilot results."""
import numpy as np
from m7_paired_seed_statistics import exact_sign_flip_p,holm
SEEDS=list(range(100129,100145))
CONDITIONS=('zero','true','mismatch','contact_state')


def summarize(successes,seed_ids):
    x=np.asarray(successes)
    if list(seed_ids)!=SEEDS or x.shape!=(16,3,2,4) or x.dtype.kind not in 'biu' or np.any((x!=0)&(x!=1)):
        raise ValueError('complete reserved-order 16 x 3 x 2 x 4 binary matrix required; unknown is not failure')
    x=x.astype(np.int64)
    low=x[:,:,0,1]-x[:,:,0,3];high=x[:,:,1,1]-x[:,:,1,3]
    differences=[high-low,high,x[:,:,1,1]-x[:,:,1,0],x[:,:,1,1]-x[:,:,1,2]]
    names=['feedback_interaction_true_vs_contact','high_true_vs_contact','high_true_vs_zero','high_true_vs_mismatch']
    boot=np.random.default_rng(98901).integers(0,16,(10000,16))
    rows=[]
    for name,per_repeat in zip(names,differences):
        d=per_repeat.sum(1)
        rows.append(dict(name=name,effect=float(d.mean()/3),per_repeat_effect=per_repeat.mean(0).tolist(),
            seed_bootstrap_95=np.quantile(d[boot].mean(1)/3,[.025,.975]).tolist(),exact_two_sided_p=exact_sign_flip_p(d)))
    for row,p in zip(rows,holm([r['exact_two_sided_p'] for r in rows])):
        row['holm_p']=p
        row['gate']=bool(row['effect']>=.1 and row['seed_bootstrap_95'][0]>0 and p<.05 and all(v>=0 for v in row['per_repeat_effect']))
    return dict(schema='candidate_feedback_confirmation_statistics_v1',seed_ids=SEEDS,
        independent_unit='environment seed; three trained repeats averaged within seed',
        contrasts=rows,feedback_interaction_supported=rows[0]['gate'],
        high_frequency_unique_tactile_benefit=all(r['gate'] for r in rows[1:]),
        execution_approved=False,technical_source_audit_required=True,
        limitation='Statistical gates alone cannot establish a claim without full pairing/source audit; non-significance is not equivalence.')

"""Predeclared duration interaction and final tactile contrasts; seed is unit."""
import numpy as np
from m7_paired_seed_statistics import exact_sign_flip_p, holm

CONDITIONS = ('zero', 'true', 'mismatch', 'contact_state')
STEPS = (720, 2880)


def summarize(successes, seed_ids):
    x = np.asarray(successes)
    seeds = list(seed_ids)
    if (x.shape != (len(seeds), 3, 2, 4) or len(set(seeds)) != len(seeds)
            or len(seeds) > 16 or x.dtype.kind not in 'biu'
            or np.any((x != 0) & (x != 1))):
        raise ValueError('complete binary seed x repeat x duration x condition required')
    if any(type(s) is not int or s not in range(100103, 100119) for s in seeds):
        raise ValueError('reserved M8 confirmation seeds required')
    x = x.astype(np.int64)
    short = x[:, :, 0, 1] - x[:, :, 0, 3]
    long = x[:, :, 1, 1] - x[:, :, 1, 3]
    differences = [long-short, long, x[:, :, 1, 1]-x[:, :, 1, 0]]
    names = ['duration_interaction_true_vs_contact', '2880_true_vs_contact', '2880_true_vs_zero']
    n = len(seeds)
    rng = np.random.default_rng(98801)
    boot = rng.integers(0, n, (10000, n)) if n else None
    results = []
    for name, per_repeat in zip(names, differences):
        d = per_repeat.sum(axis=1)
        results.append(dict(name=name,
            effect=float(d.mean()/3) if n else None,
            seed_bootstrap_95=np.quantile(d[boot].mean(1)/3, [.025, .975]).tolist() if n else [None, None],
            exact_two_sided_p=exact_sign_flip_p(d),
            per_repeat_effect=per_repeat.mean(0).tolist() if n else [None]*3))
    for item, adjusted in zip(results, holm([r['exact_two_sided_p'] for r in results])):
        item['holm_p'] = adjusted
        item['gate'] = bool(n >= 12 and item['effect'] >= .1
                            and item['seed_bootstrap_95'][0] > 0 and adjusted < .05
                            and all(v >= 0 for v in item['per_repeat_effect']))
    return dict(eligible_seed_ids=seeds, independent_unit='seed averaged over three fixed training repeats',
                conditions=CONDITIONS, steps=STEPS, successes_by_repeat_step_condition=x.sum(0).tolist(),
                contrasts=results, duration_explanation_supported=results[0]['gate'],
                final_unique_tactile_benefit=all(r['gate'] for r in results[1:]),
                assumptions='Sign-flip assumes symmetry of paired seed differences; bootstrap conditional on three fixed trained repeats. Exclusions must be reported separately. Non-significance is not equivalence.')

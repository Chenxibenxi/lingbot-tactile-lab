"""Predeclared seed-cluster statistics for 3 repetitions x 4 paired conditions."""
import numpy as np
CONDITIONS=('zero','true','mismatch','true_input_zero')

def exact_sign_flip_p(integer_seed_differences):
    d=np.asarray(integer_seed_differences)
    if d.ndim!=1 or not np.issubdtype(d.dtype,np.integer) or len(d)>16:raise ValueError('up to16 integer seed differences required')
    if not len(d):return 1.
    bits=(np.arange(2**len(d),dtype=np.uint32)[:,None]>>np.arange(len(d),dtype=np.uint32))&1
    totals=(2*bits.astype(np.int16)-1)@d
    return float(np.mean(np.abs(totals)>=abs(int(d.sum()))))

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('valid probabilities required')
    order=np.argsort(p,kind='stable');adjusted=np.empty_like(p);running=0.
    for i,j in enumerate(order):running=max(running,min(1.,(len(p)-i)*p[j]));adjusted[j]=running
    return adjusted.tolist()

def summarize_paired_success(successes,seed_ids):
    x=np.asarray(successes);seeds=list(seed_ids)
    if (x.shape!=(len(seeds),3,4) or len(set(seeds))!=len(seeds) or len(seeds)>16 or
        x.dtype.kind not in 'biu' or np.any((x!=0)&(x!=1))):raise ValueError('unique seed x 3 repeats x 4 binary complete outcomes required')
    if any(type(s) is not int or s not in range(100071,100087) for s in seeds):raise ValueError('independent fixed confirmation seeds only')
    x=x.astype(np.int64);rng=np.random.default_rng(98601);n=len(seeds);boot=rng.integers(0,n,(10000,n)) if n else None;contrasts=[]
    for j in (0,2,3):
        d=(x[:,:,1]-x[:,:,j]).sum(1)
        effect=float(d.mean()/3) if n else None
        interval=np.quantile(d[boot].mean(1)/3,[.025,.975],method='linear').tolist() if n else [None,None]
        contrasts.append({'comparison':CONDITIONS[j],'success_rate_difference':effect,'seed_cluster_bootstrap_95':interval,'exact_two_sided_p':exact_sign_flip_p(d),'per_repeat_difference':(x[:,:,1]-x[:,:,j]).mean(0).tolist() if n else [None]*3})
    for c,p in zip(contrasts,holm([c['exact_two_sided_p'] for c in contrasts])):c['holm_adjusted_p']=p
    stable=(n>=12 and contrasts[0]['success_rate_difference']>=.1 and all(c['success_rate_difference']>0 and c['seed_cluster_bootstrap_95'][0]>0 and c['holm_adjusted_p']<.05 for c in contrasts) and all(v>=0 for v in contrasts[0]['per_repeat_difference']))
    return {'eligible_seed_ids':seeds,'eligible_seed_count':n,'independent_unit':'seed, averaged over three fixed training repetitions','conditions':CONDITIONS,'successes_per_condition_repeat':x.sum(0).tolist(),'contrasts':contrasts,'predeclared_stable_benefit_criterion_met':bool(stable),'assumptions':'sign-flip uses symmetry of paired seed differences; bootstrap resamples seeds conditional on three trained repetitions; all attempted exclusions must be reported separately'}

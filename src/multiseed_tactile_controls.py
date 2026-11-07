"""Causal statistical and cross-training-episode tactile controls for M1."""
import torch
from terminal_h50_contract import tactile_arm

def payload_for(item,arm):
    payload=item['mismatch_tactile'] if arm=='mismatch' else item['tactile']
    if arm in ('true','zero'):return tactile_arm(payload,arm)
    result={k:v.clone() for k,v in payload.items()}
    if arm=='mean':
        h=result['tactile_history'];mask=result['tactile_mask']
        weight=mask.unsqueeze(-1)
        mean=(h*weight).sum(1,keepdim=True)/weight.sum(1,keepdim=True).clamp_min(torch.finfo(h.dtype).eps)
        result['tactile_history']=mean.expand_as(h).clone()
    elif arm!='mismatch':raise ValueError('unknown M1 control')
    return result

def assign_mismatches(items):
    """Donors always training, from a different seed; contact stratum preserved.

    No target/teacher/label participates in assignment. Validation never donates.
    """
    train=[x for x in items if x['meta']['fold']=='train']
    seeds=sorted({x['meta']['seed'] for x in train})
    if len(seeds)<2:raise ValueError('cross-seed control needs independent train seeds')
    for item in items:
        alternatives=[s for s in seeds if s!=item['meta']['seed']]
        target_seed=alternatives[item['meta']['frame_index']%len(alternatives)]
        group=[x for x in train if x['meta']['seed']==target_seed and x['meta']['contact']==item['meta']['contact']]
        if not group:raise ValueError('missing donor contact stratum')
        donor=group[item['meta']['frame_index']%len(group)]
        item['mismatch_tactile']={k:v.clone() for k,v in donor['tactile'].items()}
        item['mismatch_meta']={k:donor['meta'][k] for k in ('seed','frame_index','fold','contact')}
    return items

def balanced_order(train,n=360):
    import numpy as np
    rng=np.random.default_rng(97041);seeds=sorted({x['meta']['seed'] for x in train});out=[]
    groups={(s,c):[i for i,x in enumerate(train) if x['meta']['seed']==s and x['meta']['contact']==c] for s in seeds for c in (False,True)}
    if any(not g for g in groups.values()):raise ValueError('each train seed needs both strata')
    for _ in range(n):
        seed=int(rng.choice(seeds));contact=bool(rng.integers(2));out.append(int(rng.choice(groups[seed,contact])))
    return out

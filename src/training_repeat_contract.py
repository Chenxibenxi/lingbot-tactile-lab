"""Matched-arm independent training repetitions without changing base or norm buffers."""
import hashlib
import torch
import numpy as np
from tactile_adapter import TactileStateResidual

def reinitialize_adapter(adapter,seed):
    if not isinstance(adapter,TactileStateResidual) or type(seed) is not int or seed<0:raise ValueError('adapter and nonnegative integer seed required')
    if any(p.dtype!=torch.float32 for p in adapter.parameters()):raise ValueError('FP32 optimizer parameters required')
    # Only CPU RNG is changed temporarily. Do not reseed unrelated CUDA streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        fresh=TactileStateResidual(adapter.to_state.out_features,adapter.temporal.hidden_size)
    params=dict(fresh.named_parameters())
    with torch.no_grad():
        for name,p in adapter.named_parameters():p.copy_(params[name].to(device=p.device,dtype=p.dtype))
    if torch.count_nonzero(adapter.to_state.weight).item():raise ValueError('initial residual must remain zero')
    digest=hashlib.sha256()
    for name,p in adapter.named_parameters():digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return {'initialization_seed':seed,'parameter_sha256':digest.hexdigest(),'output_projection_exact_zero':True,'normalization_buffers_preserved':True}

def balanced_order(train,n,seed):
    if type(n) is not int or n<1 or type(seed) is not int or seed<0:raise ValueError('positive steps and nonnegative seed required')
    if any(x['meta']['fold']!='train' for x in train):raise ValueError('validation/test cannot enter training order')
    seeds=sorted({x['meta']['seed'] for x in train});groups={(s,c):[i for i,x in enumerate(train) if x['meta']['seed']==s and x['meta']['contact']==c] for s in seeds for c in (False,True)}
    if not seeds or any(not g for g in groups.values()):raise ValueError('each training seed needs both contact strata')
    rng=np.random.default_rng(seed);out=[]
    for _ in range(n):
        s=int(rng.choice(seeds));c=bool(rng.integers(2));out.append(int(rng.choice(groups[s,c])))
    return out

def dataset_fingerprint(items):
    """Hash transformed fixed inputs/targets as well as raw-source provenance."""
    import json
    digest=hashlib.sha256()
    for item in items:
        digest.update(json.dumps(item['meta'],sort_keys=True,default=str).encode())
        for group in ['batch','tactile','mismatch_tactile']:
            for name,tensor in sorted(item[group].items()):
                digest.update(json.dumps([group,name,str(tensor.dtype),list(tensor.shape)]).encode())
                digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(item['valid'].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

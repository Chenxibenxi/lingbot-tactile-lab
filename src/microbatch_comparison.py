"""CPU-only numerical comparison with preregistered BF16 screening thresholds."""
import hashlib
import json
import math
import torch


def tensor_digest(named):
    h=hashlib.sha256()
    for key,t in sorted(named.items()):
        t=t.detach().cpu().contiguous()
        h.update(json.dumps([key,str(t.dtype),list(t.shape)]).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def compare(reference,candidate,reference_loss,candidate_loss,limits):
    if not reference or set(reference)!=set(candidate):raise ValueError('gradient key mismatch')
    rows=[];left=[];right=[]
    for key in sorted(reference):
        a,b=reference[key],candidate[key]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('gradient shape/dtype mismatch')
        a=a.detach().cpu().double().reshape(-1);b=b.detach().cpu().double().reshape(-1)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite gradient')
        delta=float((a-b).norm());norm=float(a.norm())
        relative=delta/max(norm,limits['norm_floor'])
        rows.append(dict(parameter=key,reference_l2=norm,difference_l2=delta,relative_l2=relative,
                         passed=delta<=limits['absolute_l2']+limits['per_parameter_relative_l2']*norm))
        left.append(a);right.append(b)
    a=torch.cat(left);b=torch.cat(right)
    an=float(a.norm());bn=float(b.norm())
    if an<=limits['norm_floor'] or bn<=limits['norm_floor']:raise ValueError('degenerate total gradient')
    relative=float((a-b).norm())/an
    cosine=float(torch.dot(a,b))/(an*bn)
    if not math.isfinite(reference_loss) or not math.isfinite(candidate_loss):raise ValueError('nonfinite loss')
    loss_difference=abs(candidate_loss-reference_loss)
    loss_pass=loss_difference<=limits['loss_absolute']+limits['loss_relative']*abs(reference_loss)
    return dict(passed=all(r['passed'] for r in rows) and relative<=limits['global_relative_l2']
                and cosine>=limits['minimum_cosine'] and loss_pass,
                relative_l2=relative,cosine=cosine,loss_difference=loss_difference,loss_pass=loss_pass,parameters=rows)

"""M7 candidate: causal single-bit contact control, shared by train/inference.

Consumes only an already time-validated tactile payload. No targets, phase,
object geometry or future observations are accepted. Not integrated into M6.
"""
import torch

FIELDS={'tactile_history','tactile_mask','tactile_age_ms','tactile_valid'}

def contact_state_payload(payload):
    if set(payload)!=FIELDS:raise ValueError('exact observation-only tactile fields required')
    h,m,age,valid=(payload[k] for k in ('tactile_history','tactile_mask','tactile_age_ms','tactile_valid'))
    if h.ndim!=4 or h.shape[1:]!=(5,4,3):raise ValueError('expected B x 5 x 4 x 3 history')
    b=h.shape[0]
    if m.shape!=(b,5,4) or age.shape!=(b,) or valid.shape!=(b,) or valid.dtype!=torch.bool:raise ValueError('invalid sensor metadata shape/type')
    if not all(torch.isfinite(v).all() for v in (h,m,age)):raise ValueError('nonfinite observation')
    if (h<0).any() or (h[...,2]>1).any() or (m<0).any() or (m>1).any():raise ValueError('invalid feature bounds')
    if ((~valid)&((h.flatten(1)!=0).any(1)|(m.flatten(1)!=0).any(1)|(age!=-1))).any():raise ValueError('invalid window must use canonical fallback')
    if (valid&((age<0)|(age>200)|~(m.flatten(1)>0).any(1))).any():raise ValueError('invalid/stale valid window')
    observed=(m>0)&valid[:,None,None]
    bit=((h[...,:2]>0).any(-1)&observed).flatten(1).any(1)
    out={k:v.clone() for k,v in payload.items()}
    # One bit per window, broadcast: no force magnitude, side, or within-window
    # ordering survives. Keep coverage/age/valid identical across controls.
    out['tactile_history'].zero_()
    out['tactile_history'][...,2]=bit[:,None,None].to(h.dtype)*observed.to(h.dtype)
    return out

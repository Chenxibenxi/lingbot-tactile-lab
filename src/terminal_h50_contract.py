"""Censored terminal H50 supervision: real actions only; teacher nuisance tail.

Terminal validity never enters model observations. Missing target values are
replaced by frozen vision-only predictions generated from CURRENT observations,
not zero-filled labels. The same completion/noise is shared by all arms. This
is a pilot imputation assumption, not ground-truth post-terminal supervision.
"""
import torch

def prepare_censored_flow(actions, joint_mask, valid_steps, teacher_actions):
    if actions.ndim!=3 or actions.shape[1:]!=(50,55) or teacher_actions.shape!=actions.shape:
        raise ValueError('H50 padded model dimensions required')
    if joint_mask.shape!=actions.shape or valid_steps.shape!=actions.shape[:2] or valid_steps.dtype!=torch.bool:
        raise ValueError('explicit real-step and joint masks required')
    if torch.any(valid_steps.sum(1)<1) or torch.any(valid_steps[:,1:] & ~valid_steps[:,:-1]):
        raise ValueError('nonempty contiguous real prefix required')
    if not torch.all(joint_mask.sum(-1)==14):raise ValueError('14 actual joint dimensions required')
    mask=joint_mask.bool() & valid_steps.unsqueeze(-1)
    if not torch.isfinite(actions[mask]).all() or not torch.isfinite(teacher_actions).all():
        raise ValueError('nonfinite real labels or frozen teacher')
    # torch.where removes arbitrary/NaN missing sentinels before any model call.
    filled=torch.where(mask,actions,teacher_actions.detach())
    return filled,mask

def tactile_arm(payload,arm):
    """No force control retains architecture, mask and sensor age."""
    result={k:v.clone() for k,v in payload.items()}
    if arm=='zero':result['tactile_history'].zero_()
    elif arm=='reverse':
        result['tactile_history']=result['tactile_history'].flip(1)
        result['tactile_mask']=result['tactile_mask'].flip(1)
    elif arm!='true':raise ValueError('unknown ablation arm')
    return result

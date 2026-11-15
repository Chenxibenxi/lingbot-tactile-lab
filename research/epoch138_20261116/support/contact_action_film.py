"""Optional observation-conditioned modulation of VLA action hidden states.

Uses only the existing causal observation tokens. The zero-initialized final
projection preserves the original policy exactly at attachment time.
"""
import torch
from torch import nn

class ContactActionFiLM(nn.Module):
    def __init__(self,width,bottleneck=64):
        super().__init__()
        self.width=width
        self.score=nn.Linear(width,1)
        self.context_norm=nn.LayerNorm(width)
        self.modulation=nn.Sequential(nn.Linear(width,bottleneck),nn.SiLU(),nn.Linear(bottleneck,2*width))
        nn.init.zeros_(self.modulation[-1].weight);nn.init.zeros_(self.modulation[-1].bias)

    def forward(self,hidden,condition):
        if condition is None:return hidden
        tokens=condition['tokens'];valid=condition['padding']
        if hidden.ndim!=3 or hidden.shape[-1]!=self.width or tokens.ndim!=3 or tokens.shape[0]!=hidden.shape[0] or tokens.shape[-1]!=self.width or valid.shape!=tokens.shape[:2] or valid.dtype!=torch.bool:
            raise ValueError('action/observation shape or validity mismatch')
        if not valid.any(1).all():raise ValueError('at least one causal observation token required')
        dtype=self.score.weight.dtype
        tokens=torch.where(valid[:,:,None],tokens,0).to(dtype)
        if not torch.isfinite(tokens).all():raise ValueError('nonfinite valid observation tokens')
        logits=self.score(tokens).squeeze(-1).masked_fill(~valid,float('-inf'))
        context=(tokens*logits.softmax(-1)[:,:,None]).sum(1)
        scale,shift=self.modulation(self.context_norm(context)).chunk(2,-1)
        return hidden+(hidden*scale[:,None].to(hidden.dtype)+shift[:,None].to(hidden.dtype))

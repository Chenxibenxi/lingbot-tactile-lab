"""Opt-in low-rank adaptation of the existing VLA action velocity projection."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from contact_trainable_parameters import configure_contact_training

class ActionProjectionLoRA(nn.Module):
    def __init__(self, base, rank=8, alpha=8.0):
        super().__init__()
        if not isinstance(base,nn.Linear) or not 0<rank<=min(base.in_features,base.out_features) or alpha<=0:
            raise ValueError('linear projection, valid rank and positive alpha required')
        self.base=base
        for p in base.parameters():p.requires_grad_(False)
        self.rank=rank;self.scale=alpha/rank
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    @property
    def weight(self):
        # Keep both the normal and action_fp32 model paths on the same weights.
        return self.base.weight+(self.lora_B@self.lora_A*self.scale).to(self.base.weight.dtype)

    @property
    def bias(self):return self.base.bias

    def forward(self,x):return F.linear(x,self.weight,self.bias)


def configure_lora_contact_training(policy):
    projection=policy.model.action_out_proj
    if not isinstance(projection,ActionProjectionLoRA):raise ValueError('explicit LoRA projection required')
    groups=configure_contact_training(policy)
    projection.lora_A.requires_grad_(True);projection.lora_B.requires_grad_(True)
    groups.append(dict(name='model.action_out_proj.lora',params=[projection.lora_A,projection.lora_B]))
    selected=[p for g in groups for p in g['params']]
    if len(selected)!=len({id(p) for p in selected}):raise ValueError('duplicate parameters')
    if {id(p) for p in selected}!={id(p) for p in policy.parameters() if p.requires_grad}:raise ValueError('optimizer coverage mismatch')
    for name,p in policy.named_parameters():
        allowed=name.startswith(('model.tactile_adapter.','model.action_contact_head.')) or name in ('model.action_out_proj.lora_A','model.action_out_proj.lora_B')
        if p.requires_grad and not allowed:raise ValueError('unexpected backbone update')
    return groups

"""Opt-in capacity controls for the frozen VLA velocity output projection.

Keeps the existing base projection and LoRA delta; new residuals initialize to
zero. All paths preserve the base compute dtype, isolating capacity from the
separate FP32 sampling experiment. Original model files are never modified.
"""
import torch
from torch import nn
from torch.nn import functional as F

class ActionProjectionCapacity(nn.Module):
    def __init__(self, lora, *, full_residual=False):
        super().__init__()
        self.lora=lora
        self.capacity_bias=nn.Parameter(torch.zeros(lora.base.out_features,device=lora.base.weight.device,dtype=torch.float32))
        if full_residual:
            self.capacity_weight=nn.Parameter(torch.zeros_like(lora.base.weight,dtype=torch.float32))
        else:
            self.register_parameter('capacity_weight',None)
    @property
    def weight(self):
        w=self.lora.weight
        return w if self.capacity_weight is None else w+self.capacity_weight.to(w.dtype)
    @property
    def bias(self):
        dtype=self.lora.base.weight.dtype
        delta=self.capacity_bias.to(dtype)
        return delta if self.lora.bias is None else self.lora.bias+delta
    def forward(self,x):
        return F.linear(x,self.weight,self.bias)

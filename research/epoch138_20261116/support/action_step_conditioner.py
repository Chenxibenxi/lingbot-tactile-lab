"""Physical command step conditioning, separate from diffusion integration dt."""
import torch
from torch import nn
class ActionStepConditioner(nn.Module):
 def __init__(self,width):
  super().__init__();self.proj=nn.Linear(1,width,bias=False);nn.init.zeros_(self.proj.weight)
 def forward(self,embeddings,step_s):
  dt=torch.as_tensor(step_s,device=embeddings.device,dtype=torch.float32)
  if dt.ndim==0:dt=dt.expand(embeddings.shape[0])
  if dt.shape!=(embeddings.shape[0],) or not torch.isfinite(dt).all() or (dt<=0).any():raise ValueError('one positive physical action step per sample')
  #20ms is the original model reference; its offset stays exactly zero.
  feature=torch.log(dt/.02)[:,None].to(self.proj.weight.dtype)
  return embeddings+self.proj(feature).to(embeddings.dtype)[:,None,:]

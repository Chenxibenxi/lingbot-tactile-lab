"""Native three-interval CWMJ force tokens; no invented old sensor features."""
import torch
from torch import nn
class CWMJTemporalConditioner(nn.Module):
 def __init__(self,width):
  super().__init__();self.tactile=nn.Sequential(nn.Linear(5,width),nn.SiLU(),nn.Linear(width,width));self.proprio=nn.Linear(32,width);self.history=nn.Linear(14,width);self.time_embedding=nn.Parameter(torch.randn(3,width)*.01);self.history_embedding=nn.Parameter(torch.randn(4,width)*.01)
 def forward(self,tactile,mask,age_ms,valid,proprio,history,history_mask):
  b=tactile.shape[0]
  if tactile.shape!=(b,3,4) or mask.shape!=(b,3) or age_ms.shape!=(b,3) or valid.shape!=(b,) or proprio.shape!=(b,32) or history.shape!=(b,4,14) or history_mask.shape!=(b,4):raise ValueError('native CWMJ sensor shape')
  if mask.dtype!=torch.bool or valid.dtype!=torch.bool or history_mask.dtype!=torch.bool:raise ValueError('explicit validity required')
  live=mask&valid[:,None]
  if not torch.isfinite(tactile[live]).all() or (tactile[live]<0).any() or not torch.isfinite(age_ms[live]).all() or (age_ms[live]<0).any():raise ValueError('invalid live tactile')
  if not torch.isfinite(proprio).all() or not torch.isfinite(history[history_mask]).all():raise ValueError('invalid proprio/history')
  dtype=self.proprio.weight.dtype
  force=torch.where(live[:,:,None],tactile,0).to(dtype);age=torch.where(live,age_ms,0).to(dtype)/1000
  touch=self.tactile(torch.cat([torch.log1p(force),age[:,:,None]],-1))+self.time_embedding[None];touch=torch.where(live[:,:,None],touch,0)
  hist=self.history(torch.where(history_mask[:,:,None],history,0).to(dtype))+self.history_embedding[None];hist=torch.where(history_mask[:,:,None],hist,0)
  tokens=torch.cat([self.proprio(proprio.to(dtype))[:,None],touch,hist],1);padding=torch.cat([torch.ones(b,1,dtype=torch.bool,device=mask.device),live,history_mask],1)
  return {'tokens':tokens,'padding':padding}

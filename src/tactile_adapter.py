"""Opt-in tactile residual for LingBot-VLA's projected *state embedding*.

This module is deliberately independent of upstream LingBot files. The model
integration still needs explicit tactile arguments wired through training and
policy RPC; importing this module alone does not change the deployed model.
"""
from __future__ import annotations

import torch
from torch import nn


class TactileStateResidual(nn.Module):
    """Map [B,5,4,3] gripper history into an additive state-token residual.

    The output projection is exactly zero-initialized, so initial enabled
    outputs match the vision-only state embedding. Invalid or absent tactile
    history is bypassed even after training. No action/55D padding is abused.
    """

    def __init__(self, state_width: int, hidden_width: int = 64):
        super().__init__()
        if state_width <= 0 or hidden_width <= 0:
            raise ValueError("positive embedding widths required")
        # For each of 4 fingers: 3 force/contact features + 1 per-bin mask.
        self.temporal = nn.GRU(input_size=4 * 4, hidden_size=hidden_width,
                               batch_first=True)
        self.to_state = nn.Linear(hidden_width + 1, state_width, bias=False)
        nn.init.zeros_(self.to_state.weight)
        self.register_buffer("train_median", torch.zeros(4, 3))
        self.register_buffer("train_scale", torch.ones(4, 3))
        self.register_buffer("norm_mode", torch.tensor(0,dtype=torch.int8))  # 0=legacy; 1=positive-only
        self.register_buffer("norm_clip_abs", torch.tensor(4.,dtype=torch.float32))

    def set_normalization(self, train_stats: dict):
        """Install *training-fold-only* median/IQR buffers into the checkpoint."""
        schema=train_stats.get("schema")
        if (schema not in ("lingbot_tactile_train_norm_v1", "lingbot_tactile_contact_norm_v2") or
                train_stats.get("status")!="fit_on_train_split_only" or not train_stats.get("source_trials")):
            raise ValueError("must provide provenance-backed training-only statistics")
        if train_stats.get("training_approval") is False:
            raise ValueError("pilot normalization is not approved for tactile model training")
        med = torch.as_tensor(train_stats["median"], dtype=self.train_median.dtype,
                              device=self.train_median.device)
        scale = torch.as_tensor(train_stats["scale_iqr_floor_0p1"], dtype=self.train_scale.dtype,
                                device=self.train_scale.device)
        if med.shape != (4, 3) or scale.shape != (4, 3) or not torch.isfinite(med).all().item() or (
                not torch.isfinite(scale).all().item()) or torch.any(scale <= 0).item():
            raise ValueError("invalid tactile normalization shape or scale")
        if schema=="lingbot_tactile_contact_norm_v2":
            import math
            clip=float(train_stats.get('normalized_force_clip_abs',float('nan')))
            if (not math.isfinite(clip) or not 0<clip<=16 or
                    not torch.equal(med[:,2],torch.zeros_like(med[:,2])) or
                    not torch.equal(scale[:,2],torch.ones_like(scale[:,2]))):
                raise ValueError('invalid positive-only contact normalization contract')
        with torch.no_grad():
            self.train_median.copy_(med)
            self.train_scale.copy_(scale)
            self.norm_mode.fill_(int(schema=="lingbot_tactile_contact_norm_v2"))
            if schema=="lingbot_tactile_contact_norm_v2":self.norm_clip_abs.fill_(clip)

    def _normalized_features(self, h):
        """Mode v2 keeps absent force at exact zero; never invents contact."""
        med=self.train_median.to(dtype=h.dtype)
        scale=self.train_scale.to(dtype=h.dtype)
        if self.norm_mode.item()==1:
            force=torch.log1p(h[...,:2])
            force=torch.where(h[...,:2]>0,(force-med[:,:2])/scale[:,:2],0.)
            force=force.clamp(-float(self.norm_clip_abs.item()),float(self.norm_clip_abs.item()))
            return torch.cat([force,h[...,2:]],dim=-1)
        transformed=torch.cat([torch.log1p(h[...,:2]),h[...,2:]],dim=-1)
        return (transformed-med)/scale

    def forward(self, state_embedding: torch.Tensor,
                tactile_history: torch.Tensor | None = None,
                coverage_mask: torch.Tensor | None = None,
                age_ms: torch.Tensor | None = None,
                window_valid: torch.Tensor | None = None,
                *, enabled: bool = True) -> torch.Tensor:
        if not enabled or tactile_history is None:
            return state_embedding
        if state_embedding.ndim != 2:
            raise ValueError("state_embedding must be [B,D]")
        batch = state_embedding.shape[0]
        if state_embedding.device != self.to_state.weight.device or tactile_history.device != state_embedding.device:
            raise ValueError("state, tactile and adapter parameters must share a device")
        if coverage_mask is None or age_ms is None or window_valid is None:
            raise ValueError("tactile history requires mask, age and validity")
        if tuple(tactile_history.shape) != (batch, 5, 4, 3):
            raise ValueError("tactile_history must be [B,5,4,3]")
        if tuple(coverage_mask.shape) != (batch, 5, 4):
            raise ValueError("coverage_mask must be [B,5,4]")
        if tuple(age_ms.shape) not in ((batch,), (batch, 1)):
            raise ValueError("age_ms must be [B] or [B,1]")
        if tuple(window_valid.shape) not in ((batch,), (batch, 1)):
            raise ValueError("window_valid must be [B] or [B,1]")
        if any(x.device != state_embedding.device for x in (coverage_mask, age_ms, window_valid)):
            raise ValueError("tactile mask, age and validity must share the state device")
        if any(not torch.isfinite(x).all().item() for x in
               (tactile_history, coverage_mask, age_ms)):
            raise ValueError("nonfinite tactile data")
        if torch.any(coverage_mask < 0).item() or torch.any(coverage_mask > 1).item():
            raise ValueError("coverage mask outside [0,1]")
        valid = window_valid.reshape(batch).to(dtype=torch.bool)
        if torch.any(age_ms.reshape(batch)[valid] < 0).item():
            raise ValueError("valid tactile sample has negative age")
        if not torch.any(valid).item():
            return state_embedding
        # Keep input model-independent: do not fit dataset statistics on test
        # data. This bounded log transform is NOT physical force calibration.
        dtype = self.to_state.weight.dtype
        h = tactile_history.to(dtype=dtype)
        if torch.any(h[..., :2] < 0).item() or torch.any(h[..., 2] < 0).item() or torch.any(h[..., 2] > 1).item():
            raise ValueError("negative load or contact fraction outside [0,1]")
        # Default identity stats preserve legacy log1p; v2 uses train-fold
        # positive-bin statistics without turning absent force into a signal.
        h = self._normalized_features(h)
        mask = coverage_mask.to(dtype=dtype)
        h = h * mask.unsqueeze(-1)
        seq = torch.cat([h, mask.unsqueeze(-1)], dim=-1).reshape(batch, 5, 16)
        _, last_hidden = self.temporal(seq)
        age = torch.log1p(age_ms.reshape(batch).clamp_min(0).to(dtype=dtype))
        delta = self.to_state(torch.cat([last_hidden[-1], age.unsqueeze(-1)], dim=-1))
        gate = (valid & (coverage_mask.reshape(batch, -1).sum(dim=-1) > 0)).to(delta.dtype)
        return state_embedding + (delta * gate.unsqueeze(-1)).to(state_embedding.dtype)

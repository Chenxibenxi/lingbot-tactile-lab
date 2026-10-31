"""CPU-verifiable physics→RPC→state-token shadow path, no VLA execution.

The existing LingBot policy does NOT import this yet. Use it to establish a
strict integration contract before touching deployed model/client files.
"""
from __future__ import annotations

import torch
from tactile_rpc import attach_tactile, decode_tactile


def build_shadow_request(vision_observation, force_probe, *, enabled=False):
    if not enabled:
        return vision_observation
    window = force_probe.policy_window() if force_probe is not None else None
    return attach_tactile(vision_observation, window, enabled=True)


def shadow_state_token(state_embedding, rpc_observation, tactile_adapter):
    """Decode optional tactile fields and add zero-init residual to a state token."""
    payload = decode_tactile(rpc_observation)
    if payload is None:
        return state_embedding
    device=state_embedding.device
    history=torch.as_tensor(payload['history'],device=device).unsqueeze(0)
    mask=torch.as_tensor(payload['mask'],device=device).unsqueeze(0)
    age=torch.as_tensor(payload['age_ms'],device=device)
    valid=torch.as_tensor(payload['valid'],device=device)
    if state_embedding.shape[0] != 1:
        raise ValueError('shadow demo handles exactly one policy request')
    return tactile_adapter(state_embedding,history,mask,age,valid)

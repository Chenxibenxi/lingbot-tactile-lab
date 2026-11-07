"""Opt-in server preprocessing bridge for tactile RPC, CPU-testable only.

Existing LingBot deploy files are untouched. A future experimental deployment
must explicitly call prepare_input BEFORE feature_transform.apply (which may
silently discard unknown keys) and pass returned tensors to sample_actions.
Capability cannot be advertised without an actual trained adapter checkpoint.
"""
from __future__ import annotations
import re

import torch
from tactile_rpc import FIELDS, decode_tactile, encode_tactile_window

TENSOR_FIELDS = ("tactile_history", "tactile_mask", "tactile_age_ms", "tactile_valid")


def model_tactile_kwargs(observation, *, device, dtype, batched):
    """Fail closed on partial or malformed prepared tactile tensors.

    Legacy vision-only requests pass no new kwargs. A patched V2 model is still
    required downstream; the original sampler must reject these arguments.
    """
    present=[key in observation for key in TENSOR_FIELDS]
    if not any(present):return {}
    if not all(present):raise ValueError('incomplete prepared tactile model fields')
    h,m,age,valid=[observation[key] for key in TENSOR_FIELDS]
    if any(not isinstance(x,torch.Tensor) for x in (h,m,age,valid)):
        raise ValueError('prepared tactile fields must all be tensors')
    prefix=(h.shape[0],) if batched else ()
    if (tuple(h.shape)!=prefix+(5,4,3) or tuple(m.shape)!=prefix+(5,4) or
            tuple(age.shape)!=prefix+(1,) or tuple(valid.shape)!=prefix+(1,) or
            valid.dtype!=torch.bool):
        raise ValueError('invalid prepared tactile batch/single geometry')
    if any(not torch.isfinite(x).all().item() for x in (h,m,age)):
        raise ValueError('nonfinite prepared tactile model input')
    return {'tactile_history':h.to(device=device,dtype=dtype),
            'tactile_mask':m.to(device=device,dtype=dtype),
            'tactile_age_ms':age.to(device=device,dtype=dtype),
            'tactile_valid':valid.to(device=device)}


def capability_metadata(*, enabled: bool, model_wired: bool = False,
                        adapter_checkpoint_sha256: str | None = None,
                        repo_root=None, adapter_checkpoint_path=None,
                        deployment_manifest_path=None) -> dict:
    if not enabled:
        return {}
    if not model_wired or not isinstance(adapter_checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", adapter_checkpoint_sha256):
        raise RuntimeError("cannot advertise tactile_v1 without model wiring and adapter artifact SHA256")
    from adapter_artifact_gate import verify_deployment_artifact
    try:
        evidence=verify_deployment_artifact(repo_root,adapter_checkpoint_path,
                                            deployment_manifest_path,adapter_checkpoint_sha256)
    except Exception as exc:
        raise RuntimeError("cannot advertise tactile_v1 without verifiable approved relay training artifact") from exc
    return {"capabilities": ["tactile_v1"],
            "tactile_adapter_sha256": adapter_checkpoint_sha256,
            "tactile_base_checkpoint_sha256": evidence['base_checkpoint_sha256'],
            "tactile_schema": "lingbot_tactile_rpc_plan_v1"}


def prepare_input(raw_observation: dict, feature_transform, *, tactile_enabled=False):
    """Preserve normal visual transform and append schema-checked tensors."""
    any_tactile = any(key in raw_observation for key in FIELDS)
    if not tactile_enabled:
        if any_tactile:
            raise RuntimeError("server has tactile disabled but received tactile fields")
        return feature_transform.apply(raw_observation, policy_eval=True)
    # Decode before transform: legacy feature_transform may drop novel keys.
    payload = decode_tactile(raw_observation) if any_tactile else decode_tactile(encode_tactile_window(None))
    visual = {key: value for key, value in raw_observation.items() if key not in FIELDS}
    transformed = feature_transform.apply(visual, policy_eval=True)
    if any(key in transformed for key in TENSOR_FIELDS):
        raise ValueError("feature transform already emitted tactile keys; refusing overwrite")
    transformed = dict(transformed)
    transformed.update({"tactile_history": torch.from_numpy(payload["history"]),
                        "tactile_mask": torch.from_numpy(payload["mask"]),
                        "tactile_age_ms": torch.from_numpy(payload["age_ms"]),
                        "tactile_valid": torch.from_numpy(payload["valid"])})
    return transformed


def stack_tactile_prepared(items: list[dict]) -> dict:
    if not items:
        raise ValueError("empty tactile batch")
    for item in items:
        if any(key not in item for key in TENSOR_FIELDS):
            raise ValueError("mixed/incomplete tactile batch")
    batch = {key: torch.stack([item[key] for item in items]) for key in TENSOR_FIELDS}
    if tuple(batch["tactile_history"].shape[1:]) != (5,4,3) or tuple(batch["tactile_mask"].shape[1:]) != (5,4):
        raise ValueError("batched tactile geometry changed")
    return batch

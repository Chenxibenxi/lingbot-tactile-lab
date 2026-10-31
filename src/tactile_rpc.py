"""Opt-in tactile fields for LingBot policy RPC, restricted to causal readings.

This prepares a message contract; the existing RoboTwin client/server do not
import it yet. Call ONLY at a genuine policy model.infer request, not at every
get_obs used by demo collection or video rendering.
"""
from __future__ import annotations

import math
import numpy as np

FIELDS = ("observation.tactile.history", "observation.tactile.mask",
          "observation.tactile.age_ms", "observation.tactile.valid")
CAPABILITY = "tactile_v1"


def require_tactile_capability(metadata: dict):
    """Old servers may silently drop unknown observation keys: refuse that."""
    capabilities = metadata.get("capabilities") if isinstance(metadata, dict) else None
    if not isinstance(capabilities, (list, tuple, set)) or CAPABILITY not in capabilities:
        raise RuntimeError("server did not advertise tactile_v1; refusing fake tactile inference")


def _fallback():
    return {FIELDS[0]: np.zeros((5, 4, 3), np.float32),
            FIELDS[1]: np.zeros((5, 4), np.float32),
            FIELDS[2]: np.asarray([-1.], np.float32),
            FIELDS[3]: np.asarray([False], np.bool_)}


def encode_tactile_window(window: dict | None, *, max_age_ms: float = 200.) -> dict:
    """No future force, object name or outcome labels ever cross the RPC."""
    if window is None:
        return _fallback()
    if window.get("schema") != "phizju_tactile_window_v1":
        raise ValueError("unexpected tactile window schema")
    q = window["quality"]
    anchor = int(window["anchor_physics_step"])
    latest = q.get("last_source_step")
    dt = float(window["dt_s"])
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("invalid time calibration")
    if latest is not None and (int(latest) > anchor or int(latest) < 0):
        raise ValueError("future/invalid tactile source physics step")
    if not q.get("window_valid", False):
        return _fallback()
    if latest is None:
        raise ValueError("valid tactile window requires a source step")
    age_ms = float(q["force_age_ms"])
    if not math.isfinite(age_ms) or age_ms < 0 or not math.isclose(age_ms, (anchor-int(latest))*dt*1000, abs_tol=.02):
        raise ValueError("force age inconsistent with physics clock")
    if age_ms > max_age_ms:
        return _fallback()  # stale sensor: explicit visual-only fallback
    history = np.asarray(window["features"], dtype=np.float32)
    mask = np.asarray(window["sample_fraction_mask"], dtype=np.float32)
    if history.shape != (5, 4, 3) or mask.shape != (5, 4):
        raise ValueError("unexpected tactile feature shape")
    if not np.isfinite(history).all() or not np.isfinite(mask).all() or np.any(history < 0):
        raise ValueError("invalid tactile numeric field")
    if np.any((mask < 0) | (mask > 1)) or not np.any(mask > 0) or np.any(history[:, :, 2] > 1):
        raise ValueError("invalid tactile coverage/contact fraction")
    return {FIELDS[0]: history, FIELDS[1]: mask,
            FIELDS[2]: np.asarray([age_ms], np.float32),
            FIELDS[3]: np.asarray([True], np.bool_)}


def attach_tactile(formatted_observation: dict, window: dict | None, *, enabled=False) -> dict:
    if not enabled:
        return formatted_observation  # bit-for-bit vision-only baseline
    if any(key in formatted_observation for key in FIELDS):
        raise ValueError("refusing to overwrite existing policy fields")
    return {**formatted_observation, **encode_tactile_window(window)}


def decode_tactile(message: dict, *, max_age_ms=200.) -> dict | None:
    """Server-side schema check; None means old vision-only client."""
    found = [key in message for key in FIELDS]
    if not any(found):
        return None
    if not all(found):
        raise ValueError("partial tactile RPC payload")
    history = np.asarray(message[FIELDS[0]], np.float32)
    mask = np.asarray(message[FIELDS[1]], np.float32)
    age = np.asarray(message[FIELDS[2]], np.float32)
    valid = np.asarray(message[FIELDS[3]], np.bool_)
    if history.shape != (5,4,3) or mask.shape != (5,4) or age.shape != (1,) or valid.shape != (1,):
        raise ValueError("invalid tactile RPC dimensions")
    if not np.isfinite(history).all() or not np.isfinite(mask).all() or not np.isfinite(age).all():
        raise ValueError("nonfinite tactile RPC payload")
    if np.any(history < 0) or np.any(history[:,:,2] > 1) or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("tactile RPC force/contact/mask outside bounds")
    if not math.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("invalid server age limit")
    if not valid[0] and (np.any(history) or np.any(mask) or age[0] != -1):
        raise ValueError("invalid sensor must be explicitly zero-masked")
    if valid[0] and (age[0] < 0 or not np.any(mask > 0)):
        raise ValueError("valid tactile sample has no recent force evidence")
    if valid[0] and age[0] > max_age_ms:
        return {"history": np.zeros_like(history), "mask": np.zeros_like(mask),
                "age_ms": np.asarray([-1.], np.float32),
                "valid": np.asarray([False], np.bool_)}
    return {"history": history, "mask": mask, "age_ms": age,
            "valid": valid}

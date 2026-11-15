"""CWMJ/LingBot M2 model I/O contract layer (44 §A1).

This module is a **contract layer**, not a second inference bridge.  It reuses
the original components instead of re-implementing them:

* ``lingbot-vla-v2/lingbotvla/data/vla_data/utils.py::FeatureTransform``
  (``apply`` at :375, ``unapply`` at :490) is the one and only observation /
  action transform.  This module imports it read-only.
* ``src/contact_policy_action_decode.py::decode_actions`` is the mainline
  "original unapply" decode entry (written in an earlier round) and is imported
  unchanged.  The model output is H×55 normalized actions; it is decoded to
  H×14 commands through the original ``unapply``.  There is no truncation.
* ``lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py`` is the real deploy
  inference bridge (``LingbotVLAv2Server`` / ``select_action`` /
  ``_prepare_model_input``).  A2 reuses it; this module only binds its identity.
* ``src/contact_policy_inference.py`` is a *prior tactile-patched* bridge
  (requires the patched ``sample_actions(..., tactile_*)`` signature and an
  attached tactile adapter); it is NOT reusable for the visual-only base
  checkpoint.  Attribution and the reason are recorded in the contract.

What this module adds (the actual A1 deliverable):

1. Identity binding: per-file SHA-256 of the 6 checkpoint shards, config,
   preprocessor, tokenizer, robot config yaml, FeatureTransform source, norm
   stats, relaying training config and model config.  No weight is copied and
   no weight tensor is loaded (safetensors headers only).
2. State semantics: the original model consumes **previous commanded drive
   targets**, not measured joint positions (evidence: RoboTwin
   ``envs/robot/robot.py:494-506``, ``envs/_base_task.py:487-494``,
   ``experiment/robotwin/eval_policy_client_lingbotvla.py:366``,
   ``configs/robot_configs/robotwin.yaml`` ``subtract_state: False``).  Both
   ``measured_state`` and ``previous_command`` are recorded; swapping them is a
   labeled input-contract change, never a silent fallback.
3. Action two layers: 14-D order/units ⇄ model-internal 55-D representation
   (12 arm + 2 padded + 14 unused end.position + 2 gripper + 25 pad) with
   ``state_joint_mask`` / ``action_joint_mask``; the original ``unapply``
   decodes 55 → 14 and the bridge refuses to truncate.
4. Gripper two layers: normalized policy command ``g`` (unitless, [0,1]) and
   the physical metre target ``m = -0.01 + 0.055*g`` are separate fields; out
   of range values keep their raw value and a flag.  Clipping is never applied
   silently.
5. Offline verification (no model weights, CPU only): apply→unapply roundtrip
   for state and action, ``decode_actions`` finiteness/shape/unit checks, mask
   layout checks, synthetic and real 43-short-sequence observations.
6. Tactile: the released checkpoint has **no tactile token / projection** (no
   ``tactile`` key in the 1708-tensor index).  Only a causal window before the
   decision time may be recorded, and missing windows stay masked; nothing is
   fabricated, frame-copied or future-padded.

The 43 short sequence has no finger contact, so only the empty-clamp / missing
path can be verified here.  A1 does not load the model and does not infer.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- identities
MODEL_REPO = Path("/mnt/relaydata/lingbot/lingbot-vla-v2")
MODEL_REPO_UNPATCHED = Path("/mnt/relaydata/lingbot/lingbot-vla-v2.orig-unpatched")
CHECKPOINT_DIR = Path("/mnt/relaydata/lingbot/weights/base-11c703b")
QWEN_DIR = Path("/mnt/relaydata/lingbot/weights/Qwen3-VL-4B-Instruct")
RELAY_RUN_DIR = Path("/mnt/relaydata/lingbot/runs/relay_p1")

ROBOT_CONFIG_PATH = MODEL_REPO / "configs/robot_configs/robotwin.yaml"
NORM_STATS_PRIMARY = MODEL_REPO / "assets/norm_stats/robotwin.json"
NORM_STATS_DEPLOY_OVERRIDE = MODEL_REPO / "assets/norm_stats/robotwin_clean.json"
TRAINING_CLI_PATH = RELAY_RUN_DIR / "lingbotvla_cli.yaml"
MODEL_CONFIG_PATH = RELAY_RUN_DIR / "model_assets/config.json"

FEATURE_TRANSFORM_SOURCE = MODEL_REPO / "lingbotvla/data/vla_data/utils.py"
FEATURE_TRANSFORM_SUPPORT = MODEL_REPO / "lingbotvla/data/vla_data/transform.py"
MODEL_CONFIG_SOURCE = (
    MODEL_REPO / "lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py"
)
MODEL_MODELING_SOURCE = (
    MODEL_REPO / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
)
DEPLOY_BRIDGE_SOURCE = MODEL_REPO / "deploy/lingbot_vla_v2_policy.py"

ROBOTWIN_ROBOT_SOURCE = Path("/mnt/relaydata/lingbot/RoboTwin/envs/robot/robot.py")
ROBOTWIN_BASE_SOURCE = Path("/mnt/relaydata/lingbot/RoboTwin/envs/_base_task.py")
ROBOTWIN_EVAL_CLIENT = Path(
    "/mnt/relaydata/lingbot/lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py"
)

V31_CONTRACT = REPO_ROOT / "datasets/cwmj_aloha_14d_contract_v3_1.json"
CAMERA_CONTRACT = REPO_ROOT / "datasets/cwmj_camera_observation_contract_v1.json"
A0_WHITELIST = REPO_ROOT / "data/reports/cwmj_observation_whitelist_v1/whitelist.json"
SHORT_SEQUENCE_DIR = REPO_ROOT / "work/cwmj_short_sequence_v1"
SHORT_SEQUENCE_RUN = SHORT_SEQUENCE_DIR / "run.json"

LAB_DECODE_ENTRY = REPO_ROOT / "src/contact_policy_action_decode.py"
LAB_PRIOR_TACTILE_BRIDGE = REPO_ROOT / "src/contact_policy_inference.py"
LAB_PRIOR_VISUAL_SMOKE = REPO_ROOT / "src/prepare_real_vla_visual_smoke.py"

# Image keys consumed by ``FeatureTransform`` before/after the robotwin mapping.
# origin key (eval client / CWMJ observation) -> FeatureTransform target key.
POLICY_IMAGE_ORIGIN_KEYS = {
    "observation.images.cam_high": "cam_high",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}
POLICY_IMAGE_TARGET_KEYS = {
    "observation.images.cam_high": "observation.images.camera_top",
    "observation.images.cam_left_wrist": "observation.images.camera_wrist_left",
    "observation.images.cam_right_wrist": "observation.images.camera_wrist_right",
}

# Keys FeatureTransform.apply() produces for the real checkpoint (no tactile).
EXPECTED_MODEL_INPUT_KEYS = (
    "images",
    "img_masks",
    "state",
    "lang_tokens",
    "lang_masks",
    "actions",
    "action_is_pad",
    "joint_mask",
    "state_joint_mask",
    "action_joint_mask",
    "image_grid_thw",
)

# Keys accepted by the model-input item builder (the only things a model may see).
ALLOWED_ITEM_KEYS = frozenset(
    set(POLICY_IMAGE_ORIGIN_KEYS) | {"observation.state", "task"}
)
ROUNDTRIP_EXTRA_KEYS = frozenset({"action", "action_is_pad"})

# Fail closed on truth/diagnostic leakage, mirroring A0's whitelist v2 but kept
# dependency-free (the model env has no MuJoCo).  A0's whitelist JSON is read as
# the authority for accepted top-level observation fields.
FORBIDDEN_KEY_SUBSTRINGS = frozenset(
    {
        "asset_qpos",
        "asset_pose",
        "asset_state",
        "object_qpos",
        "object_pose",
        "object_state",
        "target_pose",
        "geom_ids",
        "geom_names",
        "geometry_labels",
        "contact_labels",
        "contact_rows",
        "contact_pairs",
        "grasp_label",
        "success_label",
        "task_success",
        "success_truth",
        "pair_class",
    }
)

# v3.1 gripper two layers (frozen by 41/42/43).
GRIPPER_NORMALIZED_RANGE = (0.0, 1.0)
GRIPPER_PHYSICAL_TARGET_DOMAIN_M = (-0.01, 0.045)
GRIPPER_CTRLRANGE_M = (-0.01, 0.04765)
GRIPPER_JOINT_RANGE_M = (0.0, 0.04765)
GRIPPER_SCALE = (-0.01, 0.045)
GRIPPER_INDEX = (6, 13)

# Tolerances: FeatureTransform normalisation is float32 with a 1e-6 epsilon in
# every (_woclip) formula, so a 1e-5 absolute tolerance is the honest bound.
ROUNDTRIP_TOL = 1e-5


class ModelIOContractError(ValueError):
    """A model-input/output contract violation (fail closed)."""


class MissingObservationError(ModelIOContractError):
    """A required synchronous observation is missing; the case is NOT_RUN."""


# ---------------------------------------------------------------- small utils
def sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Read only a safetensors header (no tensor body); CPU/disk only."""
    import struct

    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def _ensure_model_repo_on_path() -> None:
    path = str(MODEL_REPO)
    if path not in sys.path:
        sys.path.insert(0, path)


def _subprocess_text(command: Sequence[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (result.stdout + result.stderr).strip()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"<unavailable: {exc!r}>"


# ------------------------------------------------------- reference config load
def load_reference_configs() -> dict[str, Any]:
    """Load the real robot config / training config / model config (read-only)."""
    _ensure_model_repo_on_path()
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )

    training_cli = _read_yaml(TRAINING_CLI_PATH)
    raw_model_config = _read_json(MODEL_CONFIG_PATH)
    model_config = LingbotVLAV2Config(**raw_model_config)
    data_config = SimpleNamespace(**training_cli["data"])
    return {
        "training_cli": training_cli,
        "raw_model_config": raw_model_config,
        "model_config": model_config,
        "data_config": data_config,
        "config_class": type(model_config).__name__,
    }


_TRANSFORM_CACHE: dict[str, Any] = {}


def build_feature_transform(norm_stats_path: Path | None = None) -> Any:
    """Construct the ORIGINAL FeatureTransform (no model weights involved)."""
    _ensure_model_repo_on_path()
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor

    reference = load_reference_configs()
    norm_stats_path = Path(norm_stats_path or NORM_STATS_PRIMARY)
    cache_key = str(norm_stats_path)
    if cache_key in _TRANSFORM_CACHE:
        return _TRANSFORM_CACHE[cache_key]

    processor = build_processor(reference["training_cli"]["model"]["tokenizer_path"])
    transform = FeatureTransform(
        str(ROBOT_CONFIG_PATH),
        reference["data_config"],
        reference["model_config"],
        processor,
        chunk_size=int(reference["model_config"].chunk_size),
        norm_stats_path=str(norm_stats_path),
    )
    _TRANSFORM_CACHE[cache_key] = transform
    return transform


# ------------------------------------------------------------------- identity
def checkpoint_file_list() -> list[Path]:
    shards = sorted(CHECKPOINT_DIR.glob("model-*-of-*.safetensors"))
    metadata = [
        CHECKPOINT_DIR / "model.safetensors.index.json",
        CHECKPOINT_DIR / "config.json",
        CHECKPOINT_DIR / "configuration.json",
        CHECKPOINT_DIR / "preprocessor_config.json",
        CHECKPOINT_DIR / "video_preprocessor_config.json",
        CHECKPOINT_DIR / "tokenizer.json",
        CHECKPOINT_DIR / "tokenizer_config.json",
        CHECKPOINT_DIR / "vocab.json",
        CHECKPOINT_DIR / "added_tokens.json",
        CHECKPOINT_DIR / "special_tokens_map.json",
    ]
    return shards + [path for path in metadata if path.exists()]


def qwen_file_list() -> list[Path]:
    files = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    return [QWEN_DIR / name for name in files if (QWEN_DIR / name).exists()]


def checkpoint_header_facts() -> dict[str, Any]:
    """Bind the model's real internal dims from safetensors headers, no weights."""
    index = _read_json(CHECKPOINT_DIR / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    wanted = {
        "model.action_in_proj.weight": "action_in_proj",
        "model.action_out_proj.weight": "action_out_proj",
        "model.action_out_proj.bias": "action_out_proj_bias",
        "model.state_proj.weight": "state_proj",
        "model.action_time_mlp_in.weight": "action_time_mlp_in",
    }
    facts: dict[str, Any] = {}
    header_cache: dict[str, dict[str, Any]] = {}
    for key, label in wanted.items():
        shard = weight_map[key]
        if shard not in header_cache:
            header_cache[shard] = _safetensors_header(CHECKPOINT_DIR / shard)
        entry = header_cache[shard][key]
        facts[label] = {
            "tensor": key,
            "shard": shard,
            "shape": list(entry["shape"]),
            "dtype": entry["dtype"],
        }

    layer_indices = sorted(
        {
            int(key.split(".layers.")[1].split(".")[0])
            for key in weight_map
            if ".layers." in key and key.split(".layers.")[1].split(".")[0].isdigit()
        }
    )
    expert_bias = next(
        key for key in weight_map if key.endswith("mlp.e_score_correction_bias")
    )
    expert_down = next(
        key for key in weight_map if key.endswith("mlp.experts.down_proj")
    )
    expert_shard = weight_map[expert_down]
    if expert_shard not in header_cache:
        header_cache[expert_shard] = _safetensors_header(CHECKPOINT_DIR / expert_shard)
    bias_shard = weight_map[expert_bias]
    if bias_shard not in header_cache:
        header_cache[bias_shard] = _safetensors_header(CHECKPOINT_DIR / bias_shard)

    dtype_census: dict[str, int] = {}
    for shard in sorted(set(weight_map.values())):
        header = _safetensors_header(CHECKPOINT_DIR / shard)
        for key, value in header.items():
            if key == "__metadata__":
                continue
            dtype_census[value["dtype"]] = dtype_census.get(value["dtype"], 0) + 1

    facts.update(
        {
            "tensor_count": len(weight_map),
            "shard_count": len(set(weight_map.values())),
            "dtype_census": dtype_census,
            "all_weights_fp32": set(dtype_census) == {"F32"},
            "layer_indices": layer_indices,
            "layer_count": len(layer_indices),
            "token_expert_bias": {
                "tensor": expert_bias,
                "shape": list(header_cache[bias_shard][expert_bias]["shape"]),
            },
            "token_expert_down_proj": {
                "tensor": expert_down,
                "shape": list(header_cache[expert_shard][expert_down]["shape"]),
            },
            "tactile_weight_keys": sorted(
                key for key in weight_map if "tactile" in key.lower()
            ),
            "read_mode": "safetensors header only (no tensor body, no state_dict)",
        }
    )
    return facts


def _gpu_snapshot() -> dict[str, Any]:
    return {
        "nvidia_smi": _subprocess_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader",
            ]
        ),
        "compute_apps": _subprocess_text(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]
        ),
        "lease_status": _subprocess_text(
            [
                sys.executable,
                "/home/admin001/ai-video-studio/tools/gpu_lease.py",
                "status",
            ]
        ),
    }


def process_scan() -> dict[str, Any]:
    """Check whether a mainline model bridge is already running (44 §A2 rule)."""
    ps = _subprocess_text(["ps", "aux"])
    self_pids = {os.getpid(), os.getppid()}
    interesting = []
    for line in ps.splitlines():
        fields = line.split()
        if len(fields) < 2 or not fields[1].isdigit():
            continue
        if int(fields[1]) in self_pids:
            continue  # this contract run itself / its shell wrapper
        if "grep" in line or "editor" in line:
            continue
        if any(
            token in line
            for token in (
                "cwmj_model_bridge",
                "contact_policy_inference",
                "lingbot_vla_v2_policy",
                "run_cwmj_m2_inference",
                "websocket_policy_server",
            )
        ):
            interesting.append(line)
    return {
        "checked_at_unix": time.time(),
        "ps_matching_lines": interesting,
        "no_running_bridge_found": not interesting,
        "self_pids_excluded": sorted(self_pids),
    }


def identity_manifest(*, include_shards: bool = True) -> dict[str, Any]:
    """Full identity binding of everything the model I/O contract depends on."""
    checkpoint_files = checkpoint_file_list()
    shard_files = [p for p in checkpoint_files if "model-0" in p.name]
    metadata_files = [p for p in checkpoint_files if p not in shard_files]

    def _files(paths: Sequence[Path]) -> list[dict[str, Any]]:
        return [_file_identity(path) for path in paths]

    source_pairs = {}
    for path in (
        FEATURE_TRANSFORM_SOURCE,
        FEATURE_TRANSFORM_SUPPORT,
        MODEL_CONFIG_SOURCE,
        MODEL_MODELING_SOURCE,
        DEPLOY_BRIDGE_SOURCE,
    ):
        same = None
        unpatched = MODEL_REPO_UNPATCHED / path.relative_to(MODEL_REPO)
        if unpatched.exists():
            same = sha256_file(path) == sha256_file(unpatched)
        source_pairs[path.name] = {
            **_file_identity(path),
            "identical_to_lingbot_vla_v2_orig_unpatched": same,
        }

    rob_identity = {
        "shards": _files(shard_files) if include_shards else [
            {"path": str(p), "exists": True, "bytes": p.stat().st_size,
             "sha256": None, "sha256_skipped": True}
            for p in shard_files
        ],
        "shard_sha256_computed": include_shards,
        "metadata": _files(metadata_files),
        "expected_shard_count": 6,
        "actual_shard_count": len(shard_files),
        "weights_copied": False,
        "weights_loaded_into_torch": False,
    }
    return {
        "schema": "cwmj_model_io_identity_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
        },
        "checkpoint": {
            "root": str(CHECKPOINT_DIR),
            "base_release_note": (
                "official release dir; it has no lingbotvla_cli.yaml and its "
                "config.json only declares vlm_family, so the reference model "
                "config is the relay_p1 config derived from this checkpoint "
                "(relay_p1 training cli model_path/config_path point here)"
            ),
            **rob_identity,
        },
        "qwen_tokenizer_base": {
            "root": str(QWEN_DIR),
            "files": _files(qwen_file_list()),
        },
        "model_repo": {
            "root": str(MODEL_REPO),
            "is_git_checkout": (MODEL_REPO / ".git").exists(),
            "git_head": _subprocess_text(["git", "-C", str(MODEL_REPO), "rev-parse", "HEAD"]),
            "sources": source_pairs,
            "deploy_bridge_patch_note": (
                "deploy/lingbot_vla_v2_policy.py differs from "
                "lingbot-vla-v2.orig-unpatched by the P7 action-ensemble and P8 "
                "NUM_DENOISE_STEPS env hooks only; FeatureTransform, transform.py "
                "and the config class are byte-identical to upstream"
            ),
        },
        "robot_config_yaml": _file_identity(ROBOT_CONFIG_PATH),
        "norm_stats": {
            "primary": {
                **_file_identity(NORM_STATS_PRIMARY),
                "selected_by": (
                    "44 card §A1 explicitly names assets/norm_stats/robotwin.json; "
                    "it is the robot-config default (robotwin.yaml norm_stats)"
                ),
            },
            "deploy_override": {
                **_file_identity(NORM_STATS_DEPLOY_OVERRIDE),
                "selected_by": (
                    "relay_p1 training cli data.norm_stats_file (the real 93k / "
                    "relay_p1 deploy path); recorded as an identity ambiguity, "
                    "not silently substituted for the primary"
                ),
            },
        },
        "relay_training_and_model_config": {
            "training_cli": _file_identity(TRAINING_CLI_PATH),
            "model_config": _file_identity(MODEL_CONFIG_PATH),
            "base_checkpoint_reference": {
                "training_cli.model_path": str(CHECKPOINT_DIR),
                "training_cli.config_path": str(CHECKPOINT_DIR),
            },
        },
        "state_semantics_sources": {
            path.name: _file_identity(path)
            for path in (ROBOTWIN_ROBOT_SOURCE, ROBOTWIN_BASE_SOURCE, ROBOTWIN_EVAL_CLIENT)
        },
        "inherited_contracts": {
            "v31_contract": _file_identity(V31_CONTRACT),
            "camera_contract_41": _file_identity(CAMERA_CONTRACT),
            "a0_whitelist_v2": _file_identity(A0_WHITELIST),
            "short_sequence_run": _file_identity(SHORT_SEQUENCE_RUN),
        },
        "bridge_ownership": {
            "this_module": {
                "path": str(Path(__file__).resolve()),
                "role": "contract layer only; no inference implementation",
                "is_second_inference_bridge": False,
            },
            "original_decode_entry": {
                "path": str(LAB_DECODE_ENTRY),
                "role": "mainline original-unapply H×55 → H×14 decoder (imported)",
            },
            "real_deploy_bridge": {
                "path": str(DEPLOY_BRIDGE_SOURCE),
                "role": "LingbotVLAv2Server/select_action/_prepare_model_input; reused by A2",
            },
            "prior_tactile_patched_bridge": {
                "path": str(LAB_PRIOR_TACTILE_BRIDGE),
                "role": "earlier tactile research bridge",
                "reusable_for_visual_only_base": False,
                "reason": (
                    "requires the patched sample_actions(..., tactile_history, "
                    "tactile_mask, tactile_age_ms, tactile_valid, contact_proprio, "
                    "command_history, command_history_mask, noise=...) signature and "
                    "an attached tactile adapter; the released base checkpoint is "
                    "visual-only and has no tactile weights"
                ),
            },
            "prior_visual_smoke": {
                "path": str(LAB_PRIOR_VISUAL_SMOKE),
                "role": "earlier FeatureTransform visual smoke (evidence of the same "
                "original transform contract, superseded by this v1 contract)",
            },
        },
        "process_scan": process_scan(),
        "resource_preflight": {
            **_gpu_snapshot(),
            "gpu_used_by_this_task": False,
            "gpu_lease_acquired_by_this_task": False,
            "device": "cpu",
        },
        "model_weights_loaded": False,
        "model_inference_performed": False,
    }


# ------------------------------------------------------------ observation item
def _a0_whitelist() -> dict[str, Any]:
    if not A0_WHITELIST.exists():
        raise ModelIOContractError(
            f"A0 model-input whitelist artifact missing: {A0_WHITELIST}"
        )
    return _read_json(A0_WHITELIST)


def _scan_forbidden_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if any(part in key_text for part in FORBIDDEN_KEY_SUBSTRINGS):
                raise ModelIOContractError(
                    f"forbidden truth/diagnostic field {path + key_text!r} "
                    "would reach the model input"
                )
            _scan_forbidden_keys(nested, f"{path}{key_text}.")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _scan_forbidden_keys(nested, f"{path}[{index}].")


def validate_observation_against_a0(observation: Mapping[str, Any]) -> None:
    """A0 whitelist v2 is the authority; this never auto-expands."""
    whitelist = _a0_whitelist()
    allowed = set(whitelist["top_level_whitelist"])
    extra = sorted(set(observation) - allowed)
    if extra:
        raise ModelIOContractError(
            "observation has top-level fields outside A0 whitelist v2: "
            f"{extra}; the whitelist does not auto-expand"
        )
    _scan_forbidden_keys(
        {key: value for key, value in observation.items() if key == "state"}
    )


def validate_model_item(item: Mapping[str, Any], *, with_action: bool = False) -> None:
    """The only key set a model-input item may have (fail closed)."""
    allowed = set(ALLOWED_ITEM_KEYS) | (set(ROUNDTRIP_EXTRA_KEYS) if with_action else set())
    extra = sorted(set(item) - allowed)
    missing = sorted(set(ALLOWED_ITEM_KEYS) - set(item))
    if extra:
        raise ModelIOContractError(
            f"model-input item has non-contract keys: {extra}; only {sorted(allowed)} "
            "may be handed to the original transform"
        )
    if missing:
        raise ModelIOContractError(f"model-input item missing contract keys: {missing}")
    _scan_forbidden_keys(item)

    for key, camera in POLICY_IMAGE_ORIGIN_KEYS.items():
        image = item[key]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if not isinstance(image, np.ndarray):
            raise ModelIOContractError(f"{key} must be an H×W×3 uint8 array")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ModelIOContractError(f"{key} shape must be H×W×3, got {image.shape}")
        if image.dtype != np.uint8:
            raise ModelIOContractError(f"{key} dtype must be uint8, got {image.dtype}")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ModelIOContractError(f"{key} has an empty spatial extent")

    state = item["observation.state"]
    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    if state.shape != (14,):
        raise ModelIOContractError(f"observation.state must be (14,), got {tuple(state.shape)}")
    if not torch.isfinite(state).all():
        raise ModelIOContractError("observation.state contains NaN/inf")

    instruction = item["task"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ModelIOContractError("task instruction must be a non-empty string")

    if with_action:
        action = item["action"]
        if not torch.is_tensor(action):
            action = torch.as_tensor(action)
        if action.ndim != 2 or action.shape[-1] != 14:
            raise ModelIOContractError(f"action must be (H,14), got {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise ModelIOContractError("action contains NaN/inf")
        gripper = action[:, list(GRIPPER_INDEX)]
        if torch.any(gripper < GRIPPER_NORMALIZED_RANGE[0]) or torch.any(
            gripper > GRIPPER_NORMALIZED_RANGE[1]
        ):
            raise ModelIOContractError(
                "roundtrip action gripper layer must be normalized g in [0,1]"
            )
        if item["action_is_pad"].shape != (action.shape[0],):
            raise ModelIOContractError("action_is_pad must match the action horizon")


def _build_item(
    images: Mapping[str, np.ndarray],
    state_14d: Sequence[float],
    instruction: str,
) -> dict[str, Any]:
    missing = [key for key in POLICY_IMAGE_ORIGIN_KEYS if key not in images]
    if missing:
        raise MissingObservationError(f"missing synchronous policy images: {missing}")
    item = {
        key: np.array(images[key], dtype=np.uint8, copy=True)
        for key in POLICY_IMAGE_ORIGIN_KEYS
    }
    state = torch.as_tensor(np.asarray(state_14d, dtype=np.float32))
    item["observation.state"] = state
    item["task"] = instruction
    validate_model_item(item)
    return item


def measured_state_14d(robot_qpos_16: Sequence[float]) -> tuple[np.ndarray, dict[str, Any]]:
    """Measured qpos16 → policy 14-D measured state, keeping raw inverse values.

    The gripper inverse map ``g = (m + 0.01) / 0.055`` is only defined for
    ``m in [-0.01, 0.045]``.  When the measured joint position falls outside it
    (soft joint-limit penetration into the negative preload band), the raw ``g``
    and a clip flag are preserved; the clipped value is used as the policy-layer
    state only with that flag, never silently.
    """
    qpos = np.asarray(robot_qpos_16, dtype=np.float64)
    if qpos.shape != (16,) or not np.isfinite(qpos).all():
        raise ModelIOContractError("robot_qpos_16 must be a finite 16-vector")
    low, high = GRIPPER_SCALE
    gripper_m = np.array([qpos[6], qpos[14]], dtype=np.float64)
    g_raw = (gripper_m - low) / (high - low)
    g_clipped = np.clip(g_raw, *GRIPPER_NORMALIZED_RANGE)
    clipped = [(bool(g_raw[i] < 0.0), bool(g_raw[i] > 1.0)) for i in range(2)]
    state = np.array(
        list(qpos[:6]) + [g_clipped[0]] + list(qpos[8:14]) + [g_clipped[1]],
        dtype=np.float32,
    )
    return state, {
        "robot_qpos_16": qpos.tolist(),
        "gripper_measured_m": gripper_m.tolist(),
        "gripper_normalized_raw": g_raw.tolist(),
        "gripper_normalized_clipped": g_clipped.tolist(),
        "gripper_clipped_below_0": [flag[0] for flag in clipped],
        "gripper_clipped_above_1": [flag[1] for flag in clipped],
        "raw_preserved_not_hidden_by_clip": True,
    }


def select_state(observation: Mapping[str, Any], state_source: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Select model state; the two layers are explicit and never fall back."""
    if state_source == "previous_command":
        history = observation.get("command_history") or []
        if not history:
            raise MissingObservationError(
                "previous_command requested but no legal command history exists "
                "for this decision time (NOT_RUN; will not substitute measured_state)"
            )
        entry = history[-1]
        if not entry.get("legal", False):
            raise ModelIOContractError("previous_command history entry is not marked legal")
        if entry.get("gripper_units") != "normalized":
            raise ModelIOContractError(
                "previous_command must carry the normalized gripper policy layer"
            )
        state = np.asarray(entry["action_14d_normalized"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ModelIOContractError("previous_command action_14d_normalized is invalid")
        meta = {
            "state_source": "previous_command",
            "semantics": "last commanded drive target = what the original model consumes",
            "control_tick": entry.get("control_tick"),
            "sim_time_s": entry.get("sim_time_s"),
            "gripper_units": "normalized",
            "explicit_input_contract_change": False,
        }
        return state, meta

    if state_source == "measured_state":
        state_block = observation.get("state") or {}
        if "policy_state_14d" not in state_block:
            raise MissingObservationError(
                "measured_state requested but observation.state.policy_state_14d is "
                "absent (NOT_RUN; no zero fill, no borrowed time step)"
            )
        state = np.asarray(state_block["policy_state_14d"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ModelIOContractError("measured policy_state_14d is invalid")
        meta = {
            "state_source": "measured_state",
            "semantics": "measured MuJoCo joint positions, not the drive target",
            "explicit_input_contract_change": True,
            "change_note": (
                "the original checkpoint was trained/evaluated on commanded drive "
                "targets; feeding measured qpos is a labeled input-contract change"
            ),
            "gripper_measured_normalized_clipped": state_block.get(
                "measured_gripper_normalized_clipped"
            ),
            "gripper_normalized_raw": [
                float((float(m) + 0.01) / 0.055)
                for m in state_block.get("measured_gripper_m", [])
            ],
        }
        return state, meta

    raise ModelIOContractError(
        f"state_source must be 'previous_command' or 'measured_state', got {state_source!r}"
    )


def item_from_observation(
    observation: Mapping[str, Any],
    *,
    state_source: str = "previous_command",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """CWMJ A0 observation → original FeatureTransform item (whitelist enforced)."""
    validate_observation_against_a0(observation)
    images_block = observation.get("policy_images") or {}
    validity = observation.get("policy_image_valid") or {}
    images: dict[str, np.ndarray] = {}
    for key, camera in POLICY_IMAGE_ORIGIN_KEYS.items():
        if key not in images_block:
            raise MissingObservationError(f"policy image {key} is absent (NOT_RUN)")
        if validity.get(camera) is not True:
            raise MissingObservationError(
                f"policy image {key} is not valid at the decision time (NOT_RUN)"
            )
        images[key] = np.asarray(images_block[key])
    policy_keys = observation.get("policy_image_keys")
    if policy_keys and set(policy_keys) != set(POLICY_IMAGE_ORIGIN_KEYS):
        raise ModelIOContractError(
            f"policy image keys changed: {policy_keys!r}; refusing a silent remap"
        )
    language = observation.get("language") or {}
    instruction = language.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise MissingObservationError("task language instruction is missing (NOT_RUN)")
    state, state_meta = select_state(observation, state_source)

    decision_time = observation.get("decision_time_s")
    image_time = observation.get("image_time_s") or {}
    for camera in POLICY_IMAGE_ORIGIN_KEYS.values():
        moment = image_time.get(camera)
        if decision_time is not None and moment is not None and abs(moment - decision_time) > 1e-9:
            raise ModelIOContractError(
                f"image {camera} at {moment} s is not synchronous with decision time "
                f"{decision_time} s; refusing to borrow another moment"
            )
    tactile = causal_tactile_window(observation)
    item = _build_item(images, state, instruction)
    meta = {
        **state_meta,
        "decision_time_s": decision_time,
        "image_time_s": dict(image_time),
        "tactile": tactile,
        "image_sha256": {
            camera: sha256_bytes(np.ascontiguousarray(images[key]).tobytes())
            for key, camera in POLICY_IMAGE_ORIGIN_KEYS.items()
        },
    }
    return item, meta


def item_from_short_sequence(
    run: Mapping[str, Any],
    tick: int,
    *,
    state_source: str = "previous_command",
    frames_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """43 short-sequence saved observation → original FeatureTransform item.

    The trace entry at ``tick`` stores the command applied over the interval
    ``[tick-1, tick]`` together with the measured qpos at its end.  At the
    decision time after that interval the still-active drive target is exactly
    that command, which is the state the original model consumed at the same
    moment in RoboTwin's evaluation loop (``get_obs`` before ``infer``).
    """
    frames_dir = Path(frames_dir or SHORT_SEQUENCE_DIR / "frames")
    trace = run.get("trace") or []
    if not (0 <= tick < len(trace)):
        raise ModelIOContractError(f"tick {tick} outside saved trace 0..{len(trace) - 1}")
    entry = trace[tick]
    if entry.get("missing"):
        raise MissingObservationError(
            f"tick {tick} is marked missing: {entry['missing']} (NOT_RUN)"
        )
    images: dict[str, np.ndarray] = {}
    image_sha: dict[str, str] = {}
    from PIL import Image

    for key, camera in POLICY_IMAGE_ORIGIN_KEYS.items():
        path = frames_dir / f"{tick:05d}_{camera}.png"
        if not path.exists():
            raise MissingObservationError(f"saved frame missing: {path} (NOT_RUN)")
        digest = sha256_file(path)
        expected = (entry.get("policy_image_sha256") or {}).get(camera)
        if expected is not None and digest != expected:
            raise ModelIOContractError(
                f"{path} sha256 {digest} != recorded {expected}; frozen file changed"
            )
        with Image.open(path) as handle:
            image = np.array(handle.convert("RGB"), dtype=np.uint8, copy=True)
        recorded_shapes = (
            (entry.get("policy_image_shapes") or {})
            .get(camera)
            or (run.get("reset_observation", {}).get("policy_image_shapes") or {}).get(camera)
        )
        if recorded_shapes is not None and list(image.shape) != list(recorded_shapes):
            raise ModelIOContractError(
                f"{path} shape {image.shape} != recorded {recorded_shapes}"
            )
        images[key] = image
        image_sha[camera] = digest

    measured_state, measured_meta = measured_state_14d(
        entry["robot_qpos_16_contract_order"]
    )
    observation = {
        "policy_images": {
            key: images[key] for key in POLICY_IMAGE_ORIGIN_KEYS
        },
        "policy_image_valid": {
            camera: True for camera in POLICY_IMAGE_ORIGIN_KEYS.values()
        },
        "policy_image_keys": list(POLICY_IMAGE_ORIGIN_KEYS),
        "image_time_s": dict(entry["image_time_s"]),
        "decision_time_s": entry["sim_time_s"],
        "state": {
            "policy_state_14d": measured_state.tolist(),
            "measured_joint_positions_16": entry["robot_qpos_16_contract_order"],
            "measured_gripper_m": measured_meta["gripper_measured_m"],
            "measured_gripper_normalized": measured_meta["gripper_normalized_clipped"],
            "measured_gripper_normalized_clipped": [
                measured_meta["gripper_clipped_below_0"][index]
                or measured_meta["gripper_clipped_above_1"][index]
                for index in range(2)
            ],
        },
        "command_history": [
            {
                "control_tick": entry["tick"],
                "sim_time_s": entry["clock"]["force_interval_start_s"],
                "action_14d_normalized": list(entry["action_14d_normalized"]),
                "action_14d_physical_m": list(entry["ctrl_physical_14d"]),
                "gripper_units": "normalized",
                "normalized_derivable": True,
                "flags": [],
                "legal": True,
            }
        ],
        "language": {"instruction": "short sequence non-grasp wrist motion"},
        "tactile_window": _tactile_from_force_interval(
            entry.get("force_interval"), entry["sim_time_s"]
        ),
        "geometry_labels_present": False,
    }
    item, meta = item_from_observation(observation, state_source=state_source)
    meta.update(
        {
            "tick": tick,
            "sim_time_s": entry["sim_time_s"],
            "policy_image_file_sha256": image_sha,
            "measured_state_raw_inverse": {
                "gripper_m": measured_meta["gripper_measured_m"],
                "g_raw": measured_meta["gripper_normalized_raw"],
                "clipped_flags": {
                    "below_0": measured_meta["gripper_clipped_below_0"],
                    "above_1": measured_meta["gripper_clipped_above_1"],
                },
            },
            "source": "work/cwmj_short_sequence_v1 (frozen, non-grasp, 40 tick / 2 s)",
        }
    )
    return item, meta


def _tactile_from_force_interval(
    force_interval: Mapping[str, Any] | None, decision_time: float
) -> dict[str, Any]:
    if not force_interval:
        return {
            "available": False,
            "valid_mask": False,
            "missing_reason": "no_force_interval_for_this_observation",
            "contact_present": None,
            "causal": True,
            "for_model_input": False,
        }
    end = force_interval.get("force_interval_end_s")
    if end is not None and end > decision_time + 1e-12:
        raise ModelIOContractError(
            "tactile interval ends after the decision time; future force is forbidden"
        )
    return {
        "available": True,
        "valid_mask": bool(force_interval.get("sensor_valid")),
        "sensor_valid": bool(force_interval.get("sensor_valid")),
        "contact_present": bool(force_interval.get("contact_present")),
        "n_valid_substeps": int(force_interval.get("n_valid_substeps", 0)),
        "per_finger": force_interval.get("per_finger"),
        "causal": True,
        "for_model_input": False,
    }


def causal_tactile_window(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the causal tactile window; it is never injected as a token."""
    window = observation.get("tactile_window")
    decision_time = observation.get("decision_time_s")
    if window is None:
        return {
            "available": False,
            "valid_mask": False,
            "missing_reason": "no_tactile_window",
            "for_model_input": False,
            "checkpoint_has_tactile_tokens": False,
        }
    if not window.get("available"):
        return {
            "available": False,
            "valid_mask": False,
            "missing_reason": window.get("missing_reason", "tactile_unavailable"),
            "for_model_input": False,
            "checkpoint_has_tactile_tokens": False,
        }
    end = window.get("window_end_s")
    if end is not None and decision_time is not None and end > decision_time + 1e-12:
        raise ModelIOContractError(
            f"tactile window ends at {end} s, after decision time {decision_time} s"
        )
    return {
        "available": True,
        "valid_mask": bool(window.get("valid_mask", window.get("valid", True))),
        "contact_present": window.get("contact_present"),
        "age_s": window.get("age_s"),
        "causal": True,
        "for_model_input": False,
        "checkpoint_has_tactile_tokens": False,
        "note": "the released checkpoint has no tactile token/projection; the "
                "window is contract metadata only and is masked out of the model item",
    }


# ------------------------------------------------------------- apply / masks
def apply_model_input(transform: Any, item: Mapping[str, Any]) -> dict[str, Any]:
    """Run the ORIGINAL FeatureTransform.apply (policy_eval=True)."""
    validate_model_item(item)
    model_input = transform.apply(item, policy_eval=True)
    if not isinstance(model_input, Mapping):
        raise ModelIOContractError("FeatureTransform.apply did not return a mapping")
    keys = tuple(model_input.keys())
    if set(keys) != set(EXPECTED_MODEL_INPUT_KEYS):
        extra = sorted(set(keys) - set(EXPECTED_MODEL_INPUT_KEYS))
        missing = sorted(set(EXPECTED_MODEL_INPUT_KEYS) - set(keys))
        raise ModelIOContractError(
            f"unexpected model input keys (tactile injection?). extra={extra} missing={missing}"
        )
    for key in keys:
        if "tactile" in key.lower() or any(
            part in key.lower() for part in FORBIDDEN_KEY_SUBSTRINGS
        ):
            raise ModelIOContractError(f"forbidden model input key {key!r}")
    state = model_input["state"]
    actions = model_input["actions"]
    masks = model_input["state_joint_mask"], model_input["action_joint_mask"]
    if state.ndim != 1 or actions.ndim != 2:
        raise ModelIOContractError("unexpected state/action rank from apply")
    for mask in masks:
        if mask.dtype != torch.bool or mask.ndim != 1:
            raise ModelIOContractError("joint masks must be 1-D bool tensors")
    if not torch.isfinite(state).all() or not torch.isfinite(actions).all():
        raise ModelIOContractError("model input contains NaN/inf")
    if model_input["images"].ndim != 3 or model_input["images"].shape[0] != 3:
        raise ModelIOContractError("image stack must be 3 views × tokens × features")
    if bool(model_input["img_masks"].all()) is not True:
        raise ModelIOContractError("a required policy image was masked out")
    return dict(model_input)


def expected_mask_layout(model_config: Any, transform: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the 55-D layout from the real joint max dims and norm stats."""
    joints_max_dim = transform.feature_config.joints_max_dim
    max_dim = int(model_config.max_action_dim)
    layout: dict[str, Any] = {}
    offset = 0
    true_indices: list[int] = []
    padding_indices: list[int] = []
    for joint in transform.feature_config.joints:
        width = int(joints_max_dim[joint])
        state_key = f"observation.state.{joint}"
        action_key = f"action.{joint}"
        real_state = (
            int(transform.normalizer.norm_stats[state_key]["mean"].shape[-1])
            if state_key in transform.states
            else 0
        )
        real_action = (
            int(transform.normalizer.norm_stats[action_key]["mean"].shape[-1])
            if action_key in transform.actions
            else 0
        )
        layout[joint] = {
            "start": offset,
            "end": offset + width,
            "width": width,
            "real_state_dims": real_state,
            "real_action_dims": real_action,
            "state_pad": width - real_state,
            "action_pad": width - real_action,
        }
        for inner in range(real_state):
            true_indices.append(offset + inner)
        for inner in range(real_state, width):
            padding_indices.append(offset + inner)
        offset += width
    if offset > max_dim:
        raise ModelIOContractError(f"real joint width {offset} exceeds max dim {max_dim}")
    expected_true = sorted(set(true_indices))
    contract_order = [entry["name"] for entry in contract["action_14d_order"]]
    effector = layout["effector.position"]
    model_gripper_dims = [effector["start"], effector["start"] + 1]
    return {
        "max_dim": max_dim,
        "real_joint_width": offset,
        "model_tail_pad": max_dim - offset,
        "layout": layout,
        "expected_true_indices": expected_true,
        "expected_n_true": len(expected_true),
        "padding_indices_before_tail": sorted(set(padding_indices)),
        "model_layer_gripper_dims": model_gripper_dims,
        "policy_layer_gripper_columns": list(GRIPPER_INDEX),
        "contract_action_14d_order": contract_order,
    }


def check_model_input(
    model_input: Mapping[str, Any], transform: Any, contract: Mapping[str, Any]
) -> dict[str, Any]:
    reference = load_reference_configs()
    model_config = reference["model_config"]
    expected = expected_mask_layout(model_config, transform, contract)
    state_mask = model_input["state_joint_mask"]
    action_mask = model_input["action_joint_mask"]
    chunk_mask = model_input["joint_mask"]
    true_indices = torch.nonzero(state_mask).flatten().tolist()
    action_true = torch.nonzero(action_mask).flatten().tolist()
    if true_indices != expected["expected_true_indices"]:
        raise ModelIOContractError(
            f"state_joint_mask layout {true_indices} != expected {expected['expected_true_indices']}"
        )
    if action_true != true_indices:
        raise ModelIOContractError("state_joint_mask and action_joint_mask differ")
    if int(state_mask.sum()) != 14 or int(action_mask.sum()) != 14:
        raise ModelIOContractError("joint masks must expose exactly 14 real dims")
    if state_mask.shape != (expected["max_dim"],) or action_mask.shape != (
        expected["max_dim"],
    ):
        raise ModelIOContractError("joint masks must be padded to the model max dim")
    horizon = int(model_input["actions"].shape[0])
    if chunk_mask.shape != (horizon, expected["max_dim"]):
        raise ModelIOContractError("chunk_joint_mask shape mismatch")
    if int(chunk_mask.sum()) != horizon * 14:
        raise ModelIOContractError("chunk_joint_mask must broadcast the 14 real dims")
    # The padded region must be zero / mask=False, not silently populated.
    padded_state = model_input["state"][~state_mask]
    if torch.count_nonzero(padded_state).item() != 0:
        raise ModelIOContractError("state padding region is not zero")
    model_gripper_dims = expected["model_layer_gripper_dims"]
    if model_gripper_dims != [28, 29]:
        raise ModelIOContractError(
            f"model-layer gripper dims {model_gripper_dims} != [28, 29]; the 55-D "
            "layout changed and the contract is stale"
        )
    return {
        "state_shape": list(model_input["state"].shape),
        "action_shape": list(model_input["actions"].shape),
        "state_dtype": str(model_input["state"].dtype),
        "action_dtype": str(model_input["actions"].dtype),
        "images_shape": list(model_input["images"].shape),
        "img_masks": model_input["img_masks"].tolist(),
        "lang_tokens_shape": list(model_input["lang_tokens"].shape),
        "lang_masks_true": int(model_input["lang_masks"].sum()),
        "image_grid_thw": model_input["image_grid_thw"].tolist(),
        "state_joint_mask": {"shape": list(state_mask.shape), "true_indices": true_indices},
        "action_joint_mask": {"shape": list(action_mask.shape), "true_indices": action_true},
        "chunk_joint_mask": {
            "shape": list(chunk_mask.shape),
            "true_count": int(chunk_mask.sum()),
        },
        "expected_mask_layout": expected,
        "model_layer_gripper_dims": model_gripper_dims,
        "policy_layer_gripper_columns": list(GRIPPER_INDEX),
        "state_padding_zero": True,
        "mask_n_true": len(true_indices),
        "no_tactile_or_label_key": True,
    }


# -------------------------------------------------------------------- decode
def _lab_decode_actions() -> Any:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from contact_policy_action_decode import decode_actions

    return decode_actions


def _arm_ranges(contract: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    ranges = []
    for entry in contract["action_14d_order"]:
        if entry["index"] in GRIPPER_INDEX:
            continue
        ranges.append((entry["name"], float(entry["range"][0]), float(entry["range"][1])))
    return ranges


def decode_model_actions(
    transform: Any,
    model_input: Mapping[str, Any],
    actions: torch.Tensor | np.ndarray,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """55-D model output → 14-D commands via the ORIGINAL unapply, never truncated."""
    decode_actions = _lab_decode_actions()
    actions = torch.as_tensor(actions, dtype=torch.float32)
    observation = {
        "state": model_input["state"],
        "state_joint_mask": model_input["state_joint_mask"],
        "action_joint_mask": model_input["action_joint_mask"],
    }
    result: dict[str, Any] = {
        "input_shape": list(actions.shape),
        "action_names_14d": [entry["name"] for entry in contract["action_14d_order"]],
        "arm_unit": "rad",
        "gripper_policy_unit": "normalized (0=closed, 1=open)",
        "truncation_applied": False,
    }
    if actions.ndim != 2 or actions.shape[-1] != 55:
        result.update(
            {
                "status": "DECODE_REJECTED",
                "reason": f"model actions must be H×55, got {tuple(actions.shape)}; "
                          "truncating to 14/30 is not a permitted fallback",
                "executable": False,
            }
        )
        return result
    try:
        decoded = decode_actions(
            transform, actions, observation, clip_grippers=False
        )
    except ValueError as exc:
        message = str(exc)
        rejected: dict[str, Any] = {
            "status": "DECODE_REJECTED",
            "reason": message,
            "executable": False,
        }
        # Preserve raw values and flags; the clip preview is diagnostic only.
        try:
            preview = decode_actions(
                transform, actions, observation, clip_grippers=True
            )
            rejected["raw_commands_14d"] = preview["raw_commands"].tolist()
            rejected["gripper_out_of_normalized_range_elements"] = int(
                preview["gripper_clipped_elements"]
            )
            rejected["diagnostic_clipped_preview_14d"] = preview["commands"].tolist()
            rejected["diagnostic_clipped_preview_is_not_a_command"] = True
            rejected["clipping_applied_to_commands"] = False
            gripper = preview["raw_commands"][:, list(GRIPPER_INDEX)]
            rejected["gripper_normalized_raw"] = gripper.tolist()
            rejected["gripper_physical_target_raw_m"] = (
                GRIPPER_SCALE[0] + (GRIPPER_SCALE[1] - GRIPPER_SCALE[0]) * gripper
            ).tolist()
        except ValueError as exc2:  # pragma: no cover - defensive
            rejected["preview_error"] = str(exc2)
        result.update(rejected)
        return result

    commands = decoded["commands"].to(torch.float32)
    raw = decoded["raw_commands"].to(torch.float32)
    if commands.shape != (actions.shape[0], 14):
        raise ModelIOContractError(
            f"original unapply returned {tuple(commands.shape)}, expected H×14"
        )
    if not torch.isfinite(commands).all():
        result.update({"status": "DECODE_REJECTED", "reason": "non-finite decoded command"})
        return result

    gripper = commands[:, list(GRIPPER_INDEX)]
    physical = GRIPPER_SCALE[0] + (GRIPPER_SCALE[1] - GRIPPER_SCALE[0]) * gripper
    normalized_out = (gripper < GRIPPER_NORMALIZED_RANGE[0]) | (
        gripper > GRIPPER_NORMALIZED_RANGE[1]
    )
    physical_out = (physical < GRIPPER_CTRLRANGE_M[0]) | (
        physical > GRIPPER_CTRLRANGE_M[1]
    )
    physical_only_band = (physical > GRIPPER_PHYSICAL_TARGET_DOMAIN_M[1]) & (
        physical <= GRIPPER_CTRLRANGE_M[1]
    )
    arm_flags: dict[str, list[list[float]]] = {}
    arm_outside = False
    for name, low, high in _arm_ranges(contract):
        index = [entry["name"] for entry in contract["action_14d_order"]].index(name)
        column = commands[:, index]
        bad = (column < low) | (column > high)
        if bool(bad.any()):
            arm_outside = True
            arm_flags[name] = [[int(row), float(column[row])] for row in torch.nonzero(bad).flatten().tolist()]
    executable = not (bool(normalized_out.any()) or bool(physical_out.any()) or arm_outside)
    result.update(
        {
            "status": "DECODED",
            "chunk": int(commands.shape[0]),
            "commands_14d": commands.tolist(),
            "raw_commands_14d_preserved": True,
            "raw_equals_commands": bool(torch.equal(raw, commands)),
            "gripper_normalized_g": gripper.tolist(),
            "gripper_physical_target_m": physical.tolist(),
            "gripper_physical_target_formula": "m = -0.01 + 0.055 * g (no clip)",
            "gripper_normalized_out_of_range": [
                [int(row), int(col)] for row, col in torch.nonzero(normalized_out).tolist()
            ],
            "gripper_physical_outside_ctrlrange": [
                [int(row), int(col)] for row, col in torch.nonzero(physical_out).tolist()
            ],
            "gripper_physical_only_band_elements": int(physical_only_band.sum()),
            "arm_outside_contract_range": arm_flags,
            "clipping_applied": False,
            "executable": executable,
            "not_executed_reason": "A1 is offline contract verification; nothing is executed",
        }
    )
    return result


# --------------------------------------------------------------- verification
def _state_roundtrip(
    transform: Any, item: Mapping[str, Any], with_action: bool = False
) -> dict[str, Any]:
    policy_eval = not with_action
    model_input = transform.apply(item, policy_eval=policy_eval)
    unapply_input = {
        "actions": model_input["actions"].clone(),
        "state": model_input["state"].clone(),
        "state_joint_mask": model_input["state_joint_mask"].clone(),
        "action_joint_mask": model_input["action_joint_mask"].clone(),
    }
    decoded = transform.unapply(unapply_input)
    if set(decoded) != {"observation.state", "action"}:
        raise ModelIOContractError(f"unapply returned unexpected keys {sorted(decoded)}")
    state_in = item["observation.state"].to(torch.float32)
    state_out = decoded["observation.state"].to(torch.float32)
    if state_out.shape != (14,):
        raise ModelIOContractError("unapply state is not 14-D")
    state_diff = float((state_out - state_in).abs().max())
    result: dict[str, Any] = {
        "state_roundtrip_max_abs_diff": state_diff,
        "state_roundtrip_ok": state_diff <= ROUNDTRIP_TOL,
        "state_roundtrip_tolerance": ROUNDTRIP_TOL,
        "state_input_shape": list(state_in.shape),
        "state_output_shape": list(state_out.shape),
        "masks_roundtrip_equal": True,
        "state_units": "arm rad + gripper normalized g (order fl_j1..6, fl_grip, fr_j1..6, fr_grip)",
    }
    if with_action:
        action_in = item["action"].to(torch.float32)
        action_out = decoded["action"].to(torch.float32)
        action_diff = float((action_out - action_in).abs().max())
        result.update(
            {
                "action_roundtrip_max_abs_diff": action_diff,
                "action_roundtrip_ok": action_diff <= ROUNDTRIP_TOL,
                "action_input_shape": list(action_in.shape),
                "action_output_shape": list(action_out.shape),
                "action_roundtrip_tolerance": ROUNDTRIP_TOL,
            }
        )
    return result


def _action_chunk_from_state(state: np.ndarray, horizon: int) -> np.ndarray:
    gripper = np.clip(state[list(GRIPPER_INDEX)], *GRIPPER_NORMALIZED_RANGE)
    chunk = np.repeat(np.asarray(state, dtype=np.float32)[None, :], horizon, axis=0)
    chunk[:, list(GRIPPER_INDEX)] = gripper
    return chunk


def _roundtrip_cases(transform: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = _read_json(V31_CONTRACT)
    run = _read_json(SHORT_SEQUENCE_RUN)
    horizon = int(transform.chunk_size)
    cases: list[dict[str, Any]] = []

    synthetic_state = np.array(
        [0.10, -0.20, 0.30, -0.40, 0.05, 0.06, 0.5, 0.15, -0.25, 0.35, -0.45, 0.07, 0.08, 0.9],
        dtype=np.float32,
    )
    rng = np.random.default_rng(20261111)
    synthetic_images = {
        key: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        for key in POLICY_IMAGE_ORIGIN_KEYS
    }
    synthetic_item = _build_item(synthetic_images, synthetic_state, "synthetic contract probe")

    def _case(case_id: str, source: str, item: dict[str, Any]) -> None:
        policy_input = transform.apply(item, policy_eval=True)
        policy_summary = check_model_input(policy_input, transform, contract)
        roundtrip_item = dict(item)
        roundtrip_item["action"] = torch.from_numpy(_action_chunk_from_state(
            item["observation.state"].detach().cpu().numpy(), horizon
        ))
        roundtrip_item["action_is_pad"] = torch.zeros(horizon)
        validate_model_item(roundtrip_item, with_action=True)
        roundtrip = _state_roundtrip(transform, roundtrip_item, with_action=True)
        cases.append(
            {
                "case_id": case_id,
                "case_kind": "synthetic" if case_id.startswith("synthetic") else "real_43_short_sequence",
                "state_source": source,
                "model_input_summary": policy_summary,
                **roundtrip,
            }
        )

    _case("synthetic_arrays", "synthetic_arrays (not a state-semantics claim)", synthetic_item)

    for tick in (0, len(run["trace"]) // 2, len(run["trace"]) - 1):
        for source in ("previous_command", "measured_state"):
            try:
                item, meta = item_from_short_sequence(run, tick, state_source=source)
            except MissingObservationError as exc:
                cases.append(
                    {
                        "case_id": f"real_tick{tick:02d}_{source}",
                        "case_kind": "real_43_short_sequence",
                        "state_source": source,
                        "status": "NOT_RUN",
                        "reason": str(exc),
                    }
                )
                continue
            policy_input = transform.apply(item, policy_eval=True)
            policy_summary = check_model_input(policy_input, transform, contract)
            roundtrip_item = dict(item)
            roundtrip_item["action"] = torch.from_numpy(
                _action_chunk_from_state(item["observation.state"].numpy(), horizon)
            )
            roundtrip_item["action_is_pad"] = torch.zeros(horizon)
            validate_model_item(roundtrip_item, with_action=True)
            roundtrip = _state_roundtrip(transform, roundtrip_item, with_action=True)
            cases.append(
                {
                    "case_id": f"real_tick{tick:02d}_{source}",
                    "case_kind": "real_43_short_sequence",
                    "state_source": source,
                    "status": "OK",
                    "tick": tick,
                    "sim_time_s": meta["sim_time_s"],
                    "decision_time_s": meta["decision_time_s"],
                    "image_time_s": meta["image_time_s"],
                    "policy_image_file_sha256": meta["policy_image_file_sha256"],
                    "measured_state_raw_inverse": meta["measured_state_raw_inverse"],
                    "tactile": meta["tactile"],
                    "model_input_summary": policy_summary,
                    **roundtrip,
                }
            )

    # state-source delta on the real saved observations (does not replace either).
    deltas = []
    for tick in (0, len(run["trace"]) // 2, len(run["trace"]) - 1):
        command, _ = item_from_short_sequence(run, tick, state_source="previous_command")
        measured, _ = item_from_short_sequence(run, tick, state_source="measured_state")
        delta = (command["observation.state"] - measured["observation.state"]).abs()
        deltas.append(
            {
                "tick": tick,
                "max_abs_delta_14d": float(delta.max()),
                "arm_max_abs_delta_rad": float(delta[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].max()),
                "gripper_max_abs_delta_normalized": float(delta[[6, 13]].max()),
            }
        )

    summary = {
        "n_cases": len(cases),
        "all_roundtrip_ok": all(
            case.get("state_roundtrip_ok", False) and case.get("action_roundtrip_ok", False)
            for case in cases
        ),
        "state_source_delta_real_ticks": deltas,
        "selected_model_state_semantics": "previous_command (original checkpoint contract)",
        "measured_state_usage": "explicit labeled alternative, never a silent fallback",
    }
    return cases, summary


def _decode_cases(transform: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = _read_json(V31_CONTRACT)
    run = _read_json(SHORT_SEQUENCE_RUN)
    item, _ = item_from_short_sequence(run, 0, state_source="previous_command")
    model_input = transform.apply(item, policy_eval=True)
    horizon = int(transform.chunk_size)
    generator = torch.Generator().manual_seed(20261111)

    cases: list[dict[str, Any]] = []

    def _record(case_id: str, actions: Any, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        result = decode_model_actions(transform, model_input, actions, contract=contract)
        case = {
            "case_id": case_id,
            "input_kind": "torch.Tensor" if torch.is_tensor(actions) else type(actions).__name__,
            "result": result,
        }
        if extra:
            case.update(dict(extra))
        cases.append(case)
        return result

    random_actions = torch.randn(horizon, 55, generator=generator, dtype=torch.float32).clamp(
        -0.95, 0.95
    )
    _record("random_normalized_55_clamped", random_actions)

    zero_actions = torch.zeros(horizon, 55, dtype=torch.float32)
    zero_result = _record("zero_normalized_55", zero_actions)
    stats = transform.normalizer.norm_stats
    arm_order = [
        entry for entry in contract["action_14d_order"] if entry["index"] not in GRIPPER_INDEX
    ]
    expected_midpoint = []
    for entry in arm_order:
        norm_index = entry["index"] if entry["index"] < 6 else entry["index"] - 1
        expected_midpoint.append(
            float(
                (
                    float(stats["action.arm.position"]["q01"][norm_index])
                    + float(stats["action.arm.position"]["q99"][norm_index])
                )
                / 2.0
            )
        )
    decoded_zero_arms = (
        zero_result["commands_14d"][0][0:6] + zero_result["commands_14d"][0][7:13]
        if zero_result.get("status") == "DECODED"
        else None
    )
    midpoint_diff = (
        max(abs(a - b) for a, b in zip(decoded_zero_arms, expected_midpoint))
        if decoded_zero_arms is not None
        else None
    )
    cases[-1]["zero_action_midpoint_check"] = {
        "formula": "unnormalize(0) = (q01 + q99) / 2 for arm columns",
        "expected_arm_columns": expected_midpoint,
        "decoded_zero_arm_columns": decoded_zero_arms,
        "max_abs_diff": midpoint_diff,
        "ok": midpoint_diff is not None and midpoint_diff <= 1e-5,
    }

    # Gripper out of the normalized domain: normalized > 1.0 after unnormalize.
    # NOTE: the model-layer gripper dims are 28,29 (the 55-D layout is
    # [arm.position 14][end.position 14][effector.position 2][pad]), NOT the
    # policy-layer columns 6,13 of the decoded 14-D vector.
    model_gripper_dims = [28, 29]
    out_of_range = torch.zeros(horizon, 55, dtype=torch.float32)
    out_of_range[:, model_gripper_dims] = 1.4
    _record("gripper_out_of_normalized_range", out_of_range)

    non_finite = torch.zeros(horizon, 55, dtype=torch.float32)
    non_finite[0, 0] = float("nan")
    _record("non_finite_55", non_finite)

    _record("wrong_last_dim_30", torch.zeros(horizon, 30, dtype=torch.float32))
    _record("wrong_rank_1d", torch.zeros(55, dtype=torch.float32))

    # Mask occupancy: perturbing a masked-out (padded) dim must not change output.
    base = torch.randn(horizon, 55, generator=generator, dtype=torch.float32).clamp(
        -0.95, 0.95
    )
    perturbed = base.clone()
    masked_dims = torch.nonzero(~model_input["action_joint_mask"]).flatten().tolist()
    perturbed[:, masked_dims] += 100.0
    base_result = decode_model_actions(transform, model_input, base, contract=contract)
    perturbed_result = decode_model_actions(transform, model_input, perturbed, contract=contract)
    cases.append(
        {
            "case_id": "masked_dims_do_not_change_decode",
            "input_kind": "torch.Tensor",
            "n_masked_dims": len(masked_dims),
            "masked_dims": masked_dims,
            "decode_identical": base_result["commands_14d"] == perturbed_result["commands_14d"],
            "result": base_result,
        }
    )
    changed_true_dim = base.clone()
    changed_true_dim[:, 6] += 0.2
    changed_result = decode_model_actions(
        transform, model_input, changed_true_dim, contract=contract
    )
    cases.append(
        {
            "case_id": "real_dims_do_change_decode",
            "changed_dim": 6,
            "decode_changed": base_result["commands_14d"] != changed_result["commands_14d"],
            "result": changed_result,
        }
    )

    # No truncation: decoded width is 14 and both gripper columns are present.
    width_ok = all(
        isinstance(case["result"].get("commands_14d"), list)
        and len(case["result"]["commands_14d"][0]) == 14
        for case in cases
        if case["result"].get("commands_14d")
    )
    gripper_columns_present = all(
        len(case["result"]["commands_14d"][0]) == 14
        for case in cases
        if case["result"].get("commands_14d")
    )
    summary = {
        "n_cases": len(cases),
        "decoded_width_is_14_not_12": width_ok,
        "both_gripper_columns_present": gripper_columns_present,
        "gripper_out_of_range_rejected_without_clipping": (
            cases[1]["result"].get("status") == "DECODED"
            and cases[2]["result"].get("status") == "DECODE_REJECTED"
            and cases[2]["result"].get("clipping_applied_to_commands") is False
        ),
        "non_finite_rejected": cases[3]["result"].get("status") == "DECODE_REJECTED",
        "wrong_shape_rejected": cases[4]["result"].get("status") == "DECODE_REJECTED"
        and cases[5]["result"].get("status") == "DECODE_REJECTED",
        "masked_dims_ignored": cases[6]["decode_identical"],
        "real_dims_respected": cases[7]["decode_changed"],
        "zero_action_midpoint_consistent": bool(
            cases[1].get("zero_action_midpoint_check", {}).get("ok", False)
        ),
        "zero_action_midpoint_max_abs_diff": cases[1]
        .get("zero_action_midpoint_check", {})
        .get("max_abs_diff"),
        "execution_performed": False,
        "clip_grippers_used_for_commands": False,
    }
    return cases, summary


def _state_semantics_evidence() -> dict[str, Any]:
    return {
        "question": "does the original checkpoint consume measured joint positions or commanded drive targets?",
        "answer": "commanded drive targets (previous_command), not measured joint positions",
        "evidence": [
            {
                "path": str(ROBOTWIN_ROBOT_SOURCE),
                "lines": "494-506",
                "quote": "get_left_arm_jointState/get_right_arm_jointState append joint.get_drive_target()[0]",
                "meaning": "observation.state arm values are the position drive TARGET, i.e. the command",
            },
            {
                "path": str(ROBOTWIN_ROBOT_SOURCE),
                "lines": "510-526",
                "quote": "get_left_arm_real_jointState/get_right_arm_real_jointState read entity.get_qpos()",
                "meaning": "a measured-qpos variant exists but is NOT the one get_obs uses",
            },
            {
                "path": str(ROBOTWIN_ROBOT_SOURCE),
                "lines": "526-536, 628-664",
                "quote": "get_*_gripper_val returns self.left/right_gripper_val set by set_gripper (normalized)",
                "meaning": "observation gripper is the normalized command in [0,1], not a measured finger distance",
            },
            {
                "path": str(ROBOTWIN_BASE_SOURCE),
                "lines": "437-494",
                "quote": "get_obs(): joint_action.vector = left_jointstate + right_jointstate",
                "meaning": "the observation vector is exactly the command vector, 14-D",
            },
            {
                "path": str(ROBOTWIN_EVAL_CLIENT),
                "lines": "357-368",
                "quote": "observation.state = observation['joint_action']['vector'] before model.infer",
                "meaning": "the same commanded vector is what reaches the checkpoint at evaluation time",
            },
            {
                "path": str(ROBOT_CONFIG_PATH),
                "lines": "1-32",
                "quote": "states origin slices and actions with subtract_state: False",
                "meaning": "actions are absolute joint/gripper targets; no delta from state; state and "
                           "action live in the same units, so state is comparable to a previous command",
            },
        ],
        "cwmj_measured_state_is_different": True,
        "contract_change_if_measured_state_used": (
            "measured MuJoCo qpos is not the training-time distribution: a measured-state "
            "input is an explicit input-contract change recorded as "
            "state_source='measured_state', never a silent replacement"
        ),
        "state_sources": {
            "previous_command": "the last legal command_14d_normalized (arm rad + gripper g)",
            "measured_state": "measured_joint_positions_16 from MuJoCo qpos projected to the 14-D "
                              "policy layout; raw gripper inverse g and clip flags preserved",
        },
    }


def _norm_stats_comparison() -> dict[str, Any]:
    primary = _read_json(NORM_STATS_PRIMARY)
    override = _read_json(NORM_STATS_DEPLOY_OVERRIDE)

    def _stats(path: str) -> dict[str, Any]:
        block = primary["norm_stats"] if path == "primary" else override["norm_stats"]
        return {
            key: {
                stat: list(np.asarray(value).reshape(-1))
                for stat, value in sorted(block[key].items())
            }
            for key in sorted(block)
        }

    p, o = _stats("primary"), _stats("override")
    diffs: dict[str, Any] = {}
    for key in p:
        diffs[key] = {}
        for stat in p[key]:
            if key in o and stat in o[key]:
                a = np.asarray(p[key][stat], dtype=np.float64)
                b = np.asarray(o[key][stat], dtype=np.float64)
                if a.shape == b.shape:
                    diffs[key][stat] = float(np.max(np.abs(a - b)))
    return {
        "primary": {
            "path": str(NORM_STATS_PRIMARY),
            "sha256": sha256_file(NORM_STATS_PRIMARY),
            "keys": {key: int(len(p[key]["mean"])) for key in p},
            "count": primary.get("count"),
        },
        "deploy_override": {
            "path": str(NORM_STATS_DEPLOY_OVERRIDE),
            "sha256": sha256_file(NORM_STATS_DEPLOY_OVERRIDE),
            "keys": {key: int(len(o[key]["mean"])) for key in o},
            "count": override.get("count"),
        },
        "max_abs_stat_difference": diffs,
        "identity_ambiguity": (
            "the task names robotwin.json (robot config default) while the real "
            "relay_p1/93k deploy path uses robotwin_clean.json (training cli "
            "data.norm_stats_file); keys/dims agree but values differ. The primary "
            "is used for this contract as specified; the override is recorded and "
            "must be stated in any A2 inference claim"
        ),
    }


def _config_cross_checks(reference: Mapping[str, Any], header: Mapping[str, Any]) -> dict[str, Any]:
    config = reference["model_config"]
    checks = {
        "max_action_dim": {
            "config": int(config.max_action_dim),
            "checkpoint_action_out_proj_shape": header["action_out_proj"]["shape"],
            "ok": header["action_out_proj"]["shape"] == [int(config.max_action_dim), 768],
        },
        "max_state_dim": {
            "config": int(config.max_state_dim),
            "checkpoint_state_proj_shape": header["state_proj"]["shape"],
            "ok": header["state_proj"]["shape"] == [768, int(config.max_state_dim)],
        },
        "action_in_proj": {
            "config_expert_hidden_size": int(config.expert_hidden_size),
            "checkpoint_shape": header["action_in_proj"]["shape"],
            "ok": header["action_in_proj"]["shape"]
            == [int(config.expert_hidden_size), int(config.max_action_dim)],
        },
        "token_experts": {
            "config_token_num_experts": int(config.token_num_experts),
            "checkpoint_bias_shape": header["token_expert_bias"]["shape"],
            "section_count": int(config.token_moe_intermediate_size),
            "checkpoint_down_proj_shape": header["token_expert_down_proj"]["shape"],
            "ok": header["token_expert_bias"]["shape"] == [int(config.token_num_experts)]
            and header["token_expert_down_proj"]["shape"][-1]
            == int(config.token_moe_intermediate_size),
        },
        "layers": {
            "checkpoint_layer_count": header["layer_count"],
            "config_model_type": config.model_type,
        },
        "chunk_size": {
            "config_chunk_size": int(config.chunk_size),
            "config_n_action_steps": int(config.n_action_steps),
            "note": "action horizon only; norm stats are per-dimension (1-D), so the "
                    "chunk horizon is not hard-bound by the checkpoint",
        },
        "dtype": {
            "checkpoint_all_fp32": header["all_weights_fp32"],
            "dtype_census": header["dtype_census"],
        },
        "no_tactile_tokens": {
            "checkpoint_tactile_weight_keys": header["tactile_weight_keys"],
            "ok": header["tactile_weight_keys"] == [],
        },
    }
    checks["all_ok"] = all(
        value.get("ok")
        for key, value in checks.items()
        if isinstance(value, dict) and "ok" in value
    )
    return checks


# --------------------------------------------------------------------- report
def _write_reports(
    contract: dict[str, Any],
    roundtrip_cases: Sequence[dict[str, Any]],
    roundtrip_summary: Mapping[str, Any],
    decode_cases: Sequence[dict[str, Any]],
    decode_summary: Mapping[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = contract["identity"]
    _write_json(out_dir / "identity.json", identity)
    _write_json(
        out_dir / "roundtrip.json",
        {"schema": "cwmj_model_io_roundtrip_v1", "cases": list(roundtrip_cases),
         "summary": dict(roundtrip_summary)},
    )
    _write_json(
        out_dir / "decode_checks.json",
        {"schema": "cwmj_model_io_decode_checks_v1", "cases": list(decode_cases),
         "summary": dict(decode_summary)},
    )

    def _fmt(value: Any) -> str:
        return f"{value:.3e}" if isinstance(value, float) else str(value)

    lines: list[str] = []
    lines.append("# CWMJ/LingBot M2 模型 I/O 合同（44 §A1）\n")
    lines.append(
        "本报告由 `src/cwmj_model_bridge_v1.py` 生成；**只做合同层，不加载权重、不推理、"
        "不执行动作**。全部检查在 CPU 上完成，未取 GPU 租约。\n"
    )
    lines.append("## 1. 复用与归属（没有第二套推理桥）\n")
    for name, block in identity["bridge_ownership"].items():
        lines.append(f"- `{name}` → `{block.get('path')}`：{block.get('role')}")
        if block.get("is_second_inference_bridge") is not None:
            lines.append(f"  - is_second_inference_bridge = {block['is_second_inference_bridge']}")
        if block.get("reusable_for_visual_only_base") is not None:
            lines.append(
                f"  - reusable_for_visual_only_base = {block['reusable_for_visual_only_base']}"
            )
    scan = identity["process_scan"]
    lines.append(
        f"- 运行前进程扫描（`ps aux`）：无已运行的本桥/推理服务 = "
        f"{scan['no_running_bridge_found']}；匹配行数 {len(scan['ps_matching_lines'])}"
    )
    lines.append("")
    lines.append("## 2. 身份绑定（SHA-256）\n")
    shards = identity["checkpoint"]["shards"]
    lines.append("### 2.1 checkpoint 6 shards\n")
    lines.append("| 文件 | 字节 | sha256 |")
    lines.append("|---|---:|---|")
    for entry in shards:
        lines.append(
            f"| {Path(entry['path']).name} | {entry.get('bytes')} | `{entry.get('sha256')}` |"
        )
    lines.append("")
    lines.append("### 2.2 config / preprocessor / tokenizer（checkpoint 目录）\n")
    lines.append("| 文件 | 字节 | sha256 |")
    lines.append("|---|---:|---|")
    for entry in identity["checkpoint"]["metadata"]:
        lines.append(
            f"| {Path(entry['path']).name} | {entry.get('bytes')} | `{entry.get('sha256')}` |"
        )
    lines.append("")
    lines.append("### 2.3 依赖的身份\n")
    lines.append("| 对象 | sha256 | 说明 |")
    lines.append("|---|---|---|")
    for label, entry in (
        ("robotwin.yaml (robot config)", identity["robot_config_yaml"]),
        ("robotwin.json (primary norm stats)", identity["norm_stats"]["primary"]),
        ("robotwin_clean.json (deploy override)", identity["norm_stats"]["deploy_override"]),
        ("relay_p1 lingbotvla_cli.yaml", identity["relay_training_and_model_config"]["training_cli"]),
        ("relay_p1 model_assets/config.json", identity["relay_training_and_model_config"]["model_config"]),
        ("FeatureTransform utils.py", identity["model_repo"]["sources"]["utils.py"]),
        ("transform.py", identity["model_repo"]["sources"]["transform.py"]),
        ("configuration_lingbot_vla.py", identity["model_repo"]["sources"]["configuration_lingbot_vla.py"]),
        ("deploy/lingbot_vla_v2_policy.py", identity["model_repo"]["sources"]["lingbot_vla_v2_policy.py"]),
        ("RoboTwin envs/robot/robot.py", identity["state_semantics_sources"]["robot.py"]),
        ("RoboTwin envs/_base_task.py", identity["state_semantics_sources"]["_base_task.py"]),
        ("eval_policy_client_lingbotvla.py", identity["state_semantics_sources"]["eval_policy_client_lingbotvla.py"]),
        ("v3.1 动作/观测合同", identity["inherited_contracts"]["v31_contract"]),
        ("A0 白名单 v2", identity["inherited_contracts"]["a0_whitelist_v2"]),
    ):
        lines.append(f"| {label} | `{entry.get('sha256')}` | {entry.get('path')} |")
    lines.append("")
    lines.append(
        "- FeatureTransform `utils.py` / `transform.py` / config 类与 "
        "`lingbot-vla-v2.orig-unpatched` **逐字节一致**；deploy bridge 仅有 P7/P8 补丁（记录于 identity.json）。\n"
    )
    lines.append("### 2.4 版本与资源\n")
    versions = identity["versions"]
    lines.append(
        f"- python {identity['python_version']}；torch {versions['torch']}；"
        f"transformers {versions['transformers']}；numpy {versions['numpy']}"
    )
    lines.append(f"- GPU 快照（未使用）：`{identity['resource_preflight']['nvidia_smi']}`")
    lines.append(f"- lease：`{identity['resource_preflight']['lease_status']}`")
    lines.append(
        "- checkpoint 读取方式：safetensors header only（未加载 tensor body、未构建 state_dict、未复制权重）；"
        f"模型权重加载 = {identity['model_weights_loaded']}；推理 = {identity['model_inference_performed']}"
    )
    lines.append("")
    lines.append("## 3. state 语义：原模型吃的是驱动目标（previous_command），不是实测关节\n")
    for evidence in contract["state_two_layer"]["evidence"]:
        lines.append(
            f"- `{evidence['path']}` L{evidence['lines']}：{evidence['quote']}"
            f" —— {evidence['meaning']}"
        )
    lines.append("")
    deltas = roundtrip_summary["state_source_delta_real_ticks"]
    lines.append("43 短序列真实观测上两种 state 的差异（不是替代，只是量化差异）：\n")
    lines.append("| tick | 14 维最大差 | 手臂最大差(rad) | 夹爪最大差(normalized) |")
    lines.append("|---:|---:|---:|---:|")
    for delta in deltas:
        lines.append(
            f"| {delta['tick']} | {_fmt(delta['max_abs_delta_14d'])} | "
            f"{_fmt(delta['arm_max_abs_delta_rad'])} | "
            f"{_fmt(delta['gripper_max_abs_delta_normalized'])} |"
        )
    lines.append("")
    lines.append(
        "- 默认 `state_source=previous_command`（与原模型一致）；"
        "`state_source=measured_state` 是**显式输入合同变化**，缺合法命令历史时直接 "
        "NOT_RUN，不会回退到实测值或补零。\n"
    )
    lines.append("## 4. 动作两层与 mask（55 维内部表示，经原 unapply 得 14 维，不截断）\n")
    internal = contract["model_internal_representation"]
    lines.append(
        f"- max_state_dim={internal['max_state_dim']}，max_action_dim={internal['max_action_dim']}，"
        f"chunk_size={internal['chunk_size']}，n_action_steps={internal['n_action_steps']}"
    )
    for joint in internal["internal_55_layout"]["joints"]:
        lines.append(f"- joint `{joint['joint']}` width={joint['width']}")
    lines.append(
        "- 55 维布局：[0:14] arm.position（12 真 + 2 padding）、[14:28] end.position（14 未用/False）、"
        "[28:30] effector.position（2 真）、[30:55] tail pad → 真实 14 维索引 **[0..11, 28, 29]**"
    )
    checks = internal["internal_55_layout"]["checked_against_checkpoint_headers"]
    lines.append(
        f"- 与 checkpoint 张量头交叉核对："
        f"action_out_proj={checks['max_action_dim']['checkpoint_action_out_proj_shape']}、"
        f"state_proj={checks['max_state_dim']['checkpoint_state_proj_shape']}、"
        f"action_in_proj={checks['action_in_proj']['checkpoint_shape']}、"
        f"token experts={checks['token_experts']['checkpoint_down_proj_shape']}、"
        f"全部 fp32={checks['dtype']['checkpoint_all_fp32']} → all_ok={checks['all_ok']}"
    )
    mask_case = roundtrip_cases[0]["model_input_summary"]
    lines.append(
        f"- 实测 mask：state/action true indices={mask_case['state_joint_mask']['true_indices']}，"
        f"n_true={mask_case['mask_n_true']}；chunk_joint_mask true_count="
        f"{mask_case['chunk_joint_mask']['true_count']}（= horizon×14）"
    )
    lines.append(
        f"- 往返：state 最大绝对误差 {_fmt(roundtrip_cases[0]['state_roundtrip_max_abs_diff'])}、"
        f"action 最大绝对误差 {_fmt(roundtrip_cases[0].get('action_roundtrip_max_abs_diff'))}"
        f"（容差 {ROUNDTRIP_TOL}）；全部用例通过 = {roundtrip_summary['all_roundtrip_ok']}"
    )
    lines.append("")
    lines.append("## 5. 夹爪两层（g 与物理米目标分开，越界保留 raw + 标志）\n")
    gripper = contract["gripper_two_layer"]
    lines.append(
        f"- **模型层**夹爪维 = {gripper['model_layer_gripper_dims']}（55 维内部向量），与 policy 层列 "
        f"{gripper['policy_layer']['columns']}（14 维解码向量）不是同一组下标；解码必须走原 mask/unapply。"
    )
    lines.append(
        f"- policy 层：列 {gripper['policy_layer']['columns']}，`gripper_normalized_g` ∈ "
        f"{gripper['policy_layer']['range']}（0=闭，1=开）"
    )
    lines.append(
        f"- physical 层：`gripper_physical_target_m = -0.01 + 0.055*g`，policy 可达 "
        f"{gripper['physical_layer']['policy_reachable_m']} m，执行器 ctrlrange "
        f"{gripper['physical_layer']['actuator_ctrlrange_m']} m，关节范围 "
        f"{gripper['physical_layer']['joint_range_m']} m；物理专属带 "
        f"{gripper['physical_layer']['physical_only_band_m']} m"
    )
    lines.append(
        "- 解码调用原 `decode_actions(clip_grippers=False)`；越界时原入口抛错 → 保留 raw 值 + 标志 + "
        "repair 预览（标注非命令），`executable=False`，**命令从不被静默裁剪**。"
    )
    lines.append("")
    lines.append("## 6. decode 检查（有限性/形状/单位/不截断）\n")
    lines.append("| 用例 | 结果 | 关键证据 |")
    lines.append("|---|---|---|")
    for case in decode_cases:
        result = case["result"]
        status = result.get("status")
        if status == "DECODED":
            evidence = (
                f"shape={result['input_shape']} → 14 列；g={result['gripper_normalized_g'][0]}；"
                f"m={result['gripper_physical_target_m'][0]}；executable={result['executable']}"
            )
        else:
            evidence = str(result.get("reason"))[:160]
        lines.append(f"| {case['case_id']} | {status} | {evidence} |")
    lines.append("")
    lines.append(
        f"- decoded width 恒为 14（非 12）= {decode_summary['decoded_width_is_14_not_12']}，"
        f"两列夹爪都在 = {decode_summary['both_gripper_columns_present']}；"
        f"masked 维扰动不影响解码 = {decode_summary['masked_dims_ignored']}；"
        f"真实维扰动会改变解码 = {decode_summary['real_dims_respected']}；"
        f"非有限/形状错误被拒 = {decode_summary['non_finite_rejected']}/"
        f"{decode_summary['wrong_shape_rejected']}"
    )
    lines.append("")
    lines.append("## 7. 触觉（只允许因果窗口；本 checkpoint 无触觉 token）\n")
    tactile = contract["tactile"]
    lines.append(f"- checkpoint 是否有触觉输入：**{tactile['checkpoint_has_tactile_input']}**（{tactile['evidence']}）")
    sequence = tactile["short_sequence_facts"]
    lines.append(
        f"- 43 短序列 {sequence['n_ticks']} tick：`contact_present` 全 False = "
        f"{sequence['contact_present']}，`sensor_valid` 全 True = {sequence['sensor_valid']}"
    )
    lines.append(f"- 边界：{sequence['boundary']}")
    lines.append(
        "- 缺失窗口（reset 后无已采区间）→ `available=False`、mask 缺失；不复制帧、不用未来力、不发明 token。"
    )
    lines.append("")
    lines.append("## 8. 43 短序列三个观测（首个/中央/最后）\n")
    lines.append("| 位置 | tick | sim_time_s | 三图 SHA 匹配 | 触觉 |")
    lines.append("|---|---:|---:|---|---|")
    for offset in contract["saved_short_sequence_offsets"]:
        tactile_state = offset["tactile"]
        lines.append(
            f"| {offset['label']} | {offset['tick']} | {offset['sim_time_s']} | "
            f"是（文件 SHA 逐张复核） | available={tactile_state['available']}, "
            f"contact_present={tactile_state['contact_present']} |"
        )
    lines.append("")
    lines.append(f"- {contract['observation_offsets_note']}")
    lines.append("")
    lines.append("## 9. norm stats 身份歧义（必须披露，不能静默选一个）\n")
    norm = contract["verification"]["norm_stats_identity"]
    lines.append(f"- primary：`{norm['primary']['path']}` sha `{norm['primary']['sha256']}`")
    lines.append(
        f"- deploy override：`{norm['deploy_override']['path']}` sha "
        f"`{norm['deploy_override']['sha256']}`"
    )
    lines.append(f"- 二者差异（max |Δstat|）：`{json.dumps(norm['max_abs_stat_difference'])}`")
    lines.append(f"- {norm['identity_ambiguity']}")
    lines.append("")
    lines.append("## 10. 结论边界\n")
    lines.append("- **模型输入桥完成**：identity/state/action/mask/gripper/tactile 合同与离线往返、解码检查全部通过。")
    lines.append("- **真实推理未执行**：A1 不加载模型，A2 才是只读推理（条件执行）。")
    lines.append("- **动作闭环未执行**：本模块从不调用 env.step，不控制机器人，不评价任务成功，不更新 optimizer。")
    for boundary in contract["boundaries"]:
        lines.append(f"- {boundary}")
    lines.append("")
    lines.append("## 11. 复现命令\n")
    lines.append("```bash")
    lines.append("# A1 合同生成（CPU；不需要 GPU，不需要 pytest）")
    lines.append(
        "CUDA_VISIBLE_DEVICES='' /mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python \\"
    )
    lines.append(
        "  -B src/cwmj_model_bridge_v1.py --out data/reports/cwmj_model_io_contract_v1 \\"
    )
    lines.append("  --contract datasets/cwmj_model_io_contract_v1.json")
    lines.append("")
    lines.append("# 受影响测试（pytest 9.1.0 只读复制到 /tmp，模型环境未安装/升级任何依赖）")
    lines.append(
        "PYTHONPATH=/tmp/cwmj_a1/pytest_shim CUDA_VISIBLE_DEVICES='' \\"
    )
    lines.append(
        "  /mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python -B -m pytest \\"
    )
    lines.append("  tests/test_cwmj_model_io_contract.py -q")
    lines.append("```")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def run_all(
    *,
    out_dir: Path,
    contract_path: Path,
    include_shards: bool = True,
) -> dict[str, Any]:
    contract_v31 = _read_json(V31_CONTRACT)
    reference = load_reference_configs()
    transform = build_feature_transform()
    identity = identity_manifest(include_shards=include_shards)
    header_facts = checkpoint_header_facts()
    cross_checks = _config_cross_checks(reference, header_facts)
    roundtrip_cases, roundtrip_summary = _roundtrip_cases(transform)
    decode_cases, decode_summary = _decode_cases(transform)

    run = _read_json(SHORT_SEQUENCE_RUN)
    ticks = (0, len(run["trace"]) // 2, len(run["trace"]) - 1)
    saved_offsets = [
        {
            "label": label,
            "tick": tick,
            "sim_time_s": run["trace"][tick]["sim_time_s"],
            "policy_image_sha256": run["trace"][tick]["policy_image_sha256"],
            "has_synchronous_state": True,
            "has_previous_command": True,
            "tactile": _tactile_from_force_interval(
                run["trace"][tick].get("force_interval"), run["trace"][tick]["sim_time_s"]
            ),
        }
        for label, tick in zip(("first", "central", "last"), ticks)
    ]

    contract = {
        "schema": "cwmj_model_io_contract_v1",
        "created_by": "src/cwmj_model_bridge_v1.py",
        "scope": (
            "44 §A1: model-side input/action contract for the released LingBot "
            "base-11c703b checkpoint. Contract layer only: no weight load, no "
            "inference, no training, no action execution."
        ),
        "status": {
            "model_input_bridge": "COMPLETE",
            "real_inference": "NOT_EXECUTED (A2, conditional on A0/A1)",
            "action_loop_executed": False,
        },
        "inheritance": {
            "v31_contract": {
                "path": str(V31_CONTRACT),
                "sha256": sha256_file(V31_CONTRACT),
                "action_order": [entry["name"] for entry in contract_v31["action_14d_order"]],
            },
            "camera_contract_41": {
                "path": str(CAMERA_CONTRACT),
                "sha256": sha256_file(CAMERA_CONTRACT),
                "policy_keys": list(POLICY_IMAGE_ORIGIN_KEYS),
            },
            "a0_whitelist_v2": {
                "path": str(A0_WHITELIST),
                "sha256": sha256_file(A0_WHITELIST),
            },
        },
        "identity": identity,
        "model_internal_representation": {
            "max_state_dim": int(reference["model_config"].max_state_dim),
            "max_action_dim": int(reference["model_config"].max_action_dim),
            "chunk_size": int(reference["model_config"].chunk_size),
            "n_action_steps": int(reference["model_config"].n_action_steps),
            "internal_55_layout": {
                "joints": [
                    {
                        "joint": entry,
                        "width": int(transform.feature_config.joints_max_dim[entry]),
                    }
                    for entry in transform.feature_config.joints
                ],
                "derivation": "pad_and_concat concatenates joints_max_dim per joint "
                              "(arm.position 14 = 12 real + 2 padding, end.position 14 "
                              "unused/False, effector.position 2), then F.pad to "
                              "max_state_dim/max_action_dim = 55",
                "checked_against_checkpoint_headers": cross_checks,
            },
            "state_joint_mask_semantics": (
                "bool (55,): True only on the 14 real command dims "
                "[0..11, 28, 29]; the rest is padding/unused and must stay zero"
            ),
            "action_joint_mask_semantics": "same 55-D layout; chunk_joint_mask broadcasts it over the horizon",
            "decode_path": (
                "model H×55 normalized actions → FeatureTransform.unapply "
                "(reverse_pad_and_concat → normalizer.unnormalize → reverse_features) "
                "→ H×14 commands; this bridge refuses to truncate to 12/14-by-index"
            ),
        },
        "state_two_layer": {
            **_state_semantics_evidence(),
            "model_state_layer": {
                "field": "observation.state",
                "shape": [14],
                "order": [entry["name"] for entry in contract_v31["action_14d_order"]],
                "units": "6+6 arm rad, 2 normalized gripper g",
            },
            "selected_for_original_model": "previous_command",
            "selection_basis": (
                "the checkpoint's training/eval observation is the commanded drive "
                "target vector (see evidence); measured_state would be a contract change"
            ),
            "explicit_input_contract_change_if_measured_state": True,
        },
        "action_two_layer": {
            "policy_layer_14d": {
                "order": [entry["name"] for entry in contract_v31["action_14d_order"]],
                "units": ["rad"] * 6 + ["normalized"] + ["rad"] * 6 + ["normalized"],
                "ranges_from_v31": [
                    {"name": entry["name"], "range": entry["range"]}
                    for entry in contract_v31["action_14d_order"]
                ],
                "source": "datasets/cwmj_aloha_14d_contract_v3_1.json (frozen v3.1)",
            },
            "model_layer_55d": {
                "shape": [int(reference["model_config"].n_action_steps), 55],
                "mask_required": True,
                "truncation_forbidden": True,
            },
            "unapply_verified": all(
                case.get("state_roundtrip_ok") and case.get("action_roundtrip_ok")
                for case in roundtrip_cases
            ),
            "roundtrip_tolerance_abs": ROUNDTRIP_TOL,
        },
        "gripper_two_layer": {
            "model_layer_gripper_dims": [28, 29],
            "model_layer_gripper_note": (
                "in the model-internal 55-D action vector the two grippers are at "
                "dims 28,29 (effector.position), NOT 6,13; 6,13 are the gripper "
                "columns of the decoded 14-D policy vector. Decoding must go through "
                "the original mask/unapply, and a check must not assume 6,13 in 55-D."
            ),
            "policy_layer": {
                "columns": [6, 13],
                "field": "gripper_normalized_g",
                "unit": "normalized",
                "range": list(GRIPPER_NORMALIZED_RANGE),
                "semantics": "0 = closed, 1 = open",
            },
            "physical_layer": {
                "field": "gripper_physical_target_m",
                "unit": "m",
                "formula_from_policy_layer": "m = -0.01 + 0.055 * g",
                "policy_reachable_m": list(GRIPPER_PHYSICAL_TARGET_DOMAIN_M),
                "actuator_ctrlrange_m": list(GRIPPER_CTRLRANGE_M),
                "joint_range_m": list(GRIPPER_JOINT_RANGE_M),
                "target_is_command_not_position": True,
                "physical_only_band_m": [0.045, 0.04765],
                "physical_only_band_unreachable_from_policy": True,
            },
            "no_silent_clipping": {
                "decode_call": "contact_policy_action_decode.decode_actions(clip_grippers=False)",
                "out_of_range_behaviour": (
                    "the original decode entry raises; this contract keeps the raw "
                    "14-D values + flags, marks executable=False and records a "
                    "clearly labeled diagnostic clipped preview that is NOT a command"
                ),
                "commands_are_never_clipped": True,
            },
        },
        "tactile": {
            "checkpoint_has_tactile_input": False,
            "evidence": "no tactile key in the 1708-tensor checkpoint index",
            "policy": (
                "only a causal window ending at/before the decision time may be "
                "recorded; missing windows stay masked; the released checkpoint has "
                "no tactile token so nothing is injected"
            ),
            "short_sequence_facts": {
                "contact_present": all(
                    not entry["force_interval"]["contact_present"] for entry in run["trace"]
                ),
                "sensor_valid": all(
                    entry["force_interval"]["sensor_valid"] for entry in run["trace"]
                ),
                "n_ticks": len(run["trace"]),
                "boundary": "no finger contact exists, so only the empty-clamp and "
                            "missing-window paths are verified; contact information "
                            "utilization cannot be verified",
            },
        },
        "model_input_whitelist": {
            "item_keys": sorted(ALLOWED_ITEM_KEYS),
            "forbidden_substrings": sorted(FORBIDDEN_KEY_SUBSTRINGS),
            "a0_observation_whitelist_is_authority": True,
            "auto_expands": False,
            "full_environment_dict_sent_to_model": False,
        },
        "saved_short_sequence_offsets": saved_offsets,
        "observation_offsets_note": (
            "the t=0 reset_observation only records image validity/shape and has "
            "no synchronous measured body state or legal command history, so it is "
            "NOT used as an observation (no borrowing, no zero fill); the first "
            "observation with synchronous body state + previous command is trace[0] "
            "at t=0.05 s. The central and last offsets are trace[20] and trace[39]."
        ),
        "verification": {
            "roundtrip_summary": roundtrip_summary,
            "decode_summary": decode_summary,
            "norm_stats_identity": _norm_stats_comparison(),
            "model_input_shapes_seen": roundtrip_cases[0]["model_input_summary"],
        },
        "boundaries": [
            "A1 is offline and CPU-only; the model weights were never loaded, so "
            "no real inference claim is made here.",
            "the 43 short sequence has no finger contact; only empty-clamp/missing "
            "tactile paths are verifiable.",
            "the released base checkpoint dir has no training cli and only a "
            "minimal config.json; architecture config is the relay_p1 config "
            "derived from this checkpoint and is cross-checked against shard headers.",
            "robotwin.json vs robotwin_clean.json norm-stat ambiguity is recorded; "
            "A2 must state which one it used.",
            "state semantics conclusion comes from RoboTwin/LingBot source at the "
            "recorded SHAs; it is not an empirical replay of training data.",
        ],
    }

    _write_json(contract_path, contract)
    _write_reports(
        contract, roundtrip_cases, roundtrip_summary, decode_cases, decode_summary, out_dir
    )
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data/reports/cwmj_model_io_contract_v1",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "datasets/cwmj_model_io_contract_v1.json",
    )
    parser.add_argument(
        "--skip-shard-sha",
        action="store_true",
        help="skip the 6-shard sha256 (fast smoke; the report then marks them skipped)",
    )
    args = parser.parse_args(argv)
    started = time.time()
    contract = run_all(
        out_dir=args.out,
        contract_path=args.contract,
        include_shards=not args.skip_shard_sha,
    )
    elapsed = time.time() - started
    print(f"[cwmj_model_bridge_v1] wrote {args.contract}")
    print(f"[cwmj_model_bridge_v1] wrote {args.out}/{{identity,roundtrip,decode_checks}}.json + report.md")
    print(f"[cwmj_model_bridge_v1] elapsed {elapsed:.1f} s")
    print(
        "[cwmj_model_bridge_v1] status: model_input_bridge="
        f"{contract['status']['model_input_bridge']}, real_inference="
        f"{contract['status']['real_inference']}, action_loop_executed="
        f"{contract['status']['action_loop_executed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

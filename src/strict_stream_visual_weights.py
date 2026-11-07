"""Strict one-shard-at-a-time visual checkpoint copy into an ALLOCATED model.

No model is constructed here. Caller must FIRST verify a visual-only meta
model against all six checkpoint headers, allocate it on a safe device/dtype
under the GPU lease and cap memory. Never use strict=False or pickle.
Partial copies on I/O/OOM invalidate the model: discard the process.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
import torch
from safetensors import safe_open
from stream_shard_metadata import audit, REFERENCE_INDEX_SHA


@dataclass(frozen=True)
class StrictStreamResult:
    missing_keys: tuple
    unexpected_keys: tuple
    checkpoint_index_sha256: str
    copied_key_count: int
    source_tensor_bytes: int
    source_dtype: str
    destination_dtype: str
    status: str = 'strict_stream_visual_values_loaded_not_tactile_training'
    training_approval: bool = False


def stream_into_allocated_model(model, checkpoint_dir, *, available_bytes,
                                expected_index_sha=REFERENCE_INDEX_SHA,
                                margin_bytes=2*1024**3,expected_shard_manifest=None):
    """Verify complete index+shape before writing ANY parameter, then stream-copy.

    This is a code contract, not enough to prove *full* GPU memory, optimizer,
    or accuracy. The baseline source, filesystem and full shard hashes must be
    pinned by the caller; returned strict result may be given to zero adapter
    attachment only after every key was copied successfully.
    """
    checkpoint_dir=Path(checkpoint_dir).resolve(strict=True)
    if not isinstance(model,torch.nn.Module) or not model.state_dict():
        raise ValueError('nonempty allocated visual-only torch model required')
    if getattr(getattr(model,'config',None),'tactile_enabled',False) or getattr(model,'tactile_adapter',None) is not None:
        raise ValueError('cannot stream old visual weights into pre-attached tactile model')
    state=model.state_dict()
    if any(tensor.is_meta for tensor in state.values()):
        raise ValueError('allocate model under bounded device/dtype before stream copying')
    if (any(tensor.device!=next(iter(state.values())).device for tensor in state.values()) or
            len({tensor.dtype for tensor in state.values()})!=1):
        raise ValueError('state_dict must share a single device and floating dtype')
    model._strict_stream_partial_failure=True  # cleared ONLY after all shards copied
    device=next(iter(state.values())).device
    if device.type=='cuda' and torch.cuda.mem_get_info(device)[0]<4*1024**3:
        raise ValueError('insufficient post-allocation free GPU memory for safe shard copy')
    header=audit(checkpoint_dir,available_bytes=available_bytes,
                 expected_index_sha=expected_index_sha,margin_bytes=margin_bytes)
    if not header['single_shard_ram_lower_bound_pass']:
        raise ValueError('one-shard host memory gate refused visual weight copying')
    mapping=json.loads((checkpoint_dir/'model.safetensors.index.json').read_text())['weight_map']
    if expected_index_sha==REFERENCE_INDEX_SHA and expected_shard_manifest is None:
        raise ValueError('original 154800 weight values require pinned six-shard SHA manifest')
    shards=None
    if expected_shard_manifest is not None:
        if (not isinstance(expected_shard_manifest,dict) or
                expected_shard_manifest.get('checkpoint_index_sha256')!=expected_index_sha):
            raise ValueError('pinned original shard manifest index mismatch')
        shards=expected_shard_manifest.get('shard_files')
        if (not isinstance(shards,dict) or set(shards)!=set(mapping.values()) or
                any(not isinstance(item,dict) or type(item.get('file_bytes')) is not int or
                    not isinstance(item.get('sha256'),str) or
                    re.fullmatch(r'[a-f0-9]{64}',item['sha256']) is None
                    for item in shards.values())):
            raise ValueError('pinned six-shard byte manifest invalid')
    if set(state)!=set(mapping):
        raise ValueError('strict visual checkpoint keys do not match allocated model')
    # Preflight EVERY shape/dtype before mutating any parameter. This original
    # source is FP32-only: refuse unsupported integer buffers or implicit casts.
    expected=set(mapping)
    for filename in sorted(set(mapping.values())):
        with safe_open(str(checkpoint_dir/filename),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                slice_=reader.get_slice(key)
                if tuple(slice_.get_shape())!=tuple(state[key].shape):
                    raise ValueError(f'checkpoint tensor shape mismatch: {key}')
                kind=slice_.get_dtype()
                dst=state[key].dtype
                if not (kind=='F32' and dst in (torch.float32,torch.bfloat16)):
                    raise ValueError(f'unsupported strict source/destination dtype: {key}')
    copied=set()
    try:
        with torch.no_grad():
            for filename in sorted(set(mapping.values())):
                path=checkpoint_dir/filename
                before=path.stat()
                if shards is not None:
                    info=shards[filename]
                    if before.st_size!=info['file_bytes']:
                        raise ValueError('original shard size changed since pinned SHA')
                    sha=hashlib.sha256()
                    with path.open('rb',buffering=1024*1024) as handle:
                        for block in iter(lambda:handle.read(4*1024*1024),b''):
                            sha.update(block)
                    if sha.hexdigest()!=info['sha256'] or path.stat().st_mtime_ns!=before.st_mtime_ns:
                        raise ValueError('original shard SHA/mtime differs before tensor copy')
                with safe_open(str(path),framework='pt',device='cpu') as reader:
                    for key in reader.keys():
                        if key not in expected or key in copied:
                            raise ValueError('unexpected or duplicate source tensor during strict stream')
                        tensor=reader.get_tensor(key)
                        state[key].copy_(tensor,non_blocking=False)
                        copied.add(key)
                        del tensor
                if path.stat().st_mtime_ns!=before.st_mtime_ns:
                    raise ValueError('original shard changed during tensor copy')
    except Exception:
        # The model may have been PARTLY copied: it is never safe to continue.
        model._strict_stream_partial_failure=True
        raise
    if copied!=expected:
        raise ValueError('incomplete original visual checkpoint copy')
    model._strict_stream_partial_failure=False
    return StrictStreamResult((),(),header['index_sha256'],len(copied),
                              header['declared_tensor_bytes'],'F32',str(next(iter(state.values())).dtype))

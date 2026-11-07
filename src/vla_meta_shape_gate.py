"""Strict model state-dict KEY/SHAPE audit against checkpoint HEADERS only.

Requires a metadata-only model (torch.device('meta')); refuses any allocated
parameter/buffer and any pre-attached tactile adapter. No value tensors,
checkpoint loading, GPU output, or training are produced here.
"""
from __future__ import annotations
import json
from pathlib import Path
from safetensors import safe_open
from stream_shard_metadata import audit,REFERENCE_INDEX_SHA


def compare_meta_model(model, checkpoint_dir, *, available_bytes,expected_index_sha=REFERENCE_INDEX_SHA):
    checkpoint_dir=Path(checkpoint_dir).resolve(strict=True)
    if getattr(getattr(model,'config',None),'tactile_enabled',False) or getattr(model,'tactile_adapter',None) is not None:
        raise ValueError('meta checkpoint key audit requires the original visual model before adapter')
    expected=model.state_dict()
    if not expected or any(not tensor.is_meta for tensor in expected.values()):
        raise ValueError('model state must be entirely meta tensors; no allocated weights allowed')
    header=audit(checkpoint_dir,available_bytes=available_bytes,expected_index_sha=expected_index_sha)
    weight_map=json.loads((checkpoint_dir/'model.safetensors.index.json').read_text())['weight_map']
    if set(expected)!=set(weight_map):
        raise ValueError(f'strict original checkpoint keys differ from meta model: missing={len(set(expected)-set(weight_map))}, extra={len(set(weight_map)-set(expected))}')
    count=0
    for filename in sorted(set(weight_map.values())):
        with safe_open(str(checkpoint_dir/filename),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                if tuple(reader.get_slice(key).get_shape())!=tuple(expected[key].shape):
                    raise ValueError(f'original checkpoint tensor shape differs: {key}')
                count+=1
    if count!=header['tensor_key_count']:
        raise AssertionError('full metadata key enumeration differs from model state')
    return {'schema':'lingbot_real_v2_meta_state_header_gate_v1',
            'status':'all_original_visual_keys_and_shapes_match_headers_only',
            'index_sha256':header['index_sha256'],'matched_key_count':count,
            'model_is_meta_only':True,'values_loaded':False,'training_approval':False,
            'single_shard_ram_lower_bound_pass':header['single_shard_ram_lower_bound_pass'],
            'warning':'No checkpoint tensor values were loaded; no actual VLA forward/inference/gradients or GPU fit proved.'}

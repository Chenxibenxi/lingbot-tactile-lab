"""Verify legacy checkpoint shard INDEX and safetensors HEADERS without tensor loads.

This is a CPU-only prerequisite for designing a safe one-shard-at-a-time
strict loader; metadata verification is NOT loading VLA, training, or GPU proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from collections import Counter
from safetensors import safe_open

REFERENCE_INDEX_SHA='62dbe509bd476ff354e6a9bea5476bfe451471a5e1160d24abb594a35fe2ec40'
DTYPE_BYTES={'F32':4,'F64':8,'F16':2,'BF16':2,'I8':1,'U8':1,'BOOL':1,'I16':2,'I32':4,'I64':8}
MARGIN=2*1024**3


def audit(directory,*,available_bytes,expected_index_sha=None,margin_bytes=MARGIN):
    directory=Path(directory).resolve(strict=True)
    if (type(available_bytes) is not int or available_bytes<=0 or
            type(margin_bytes) is not int or margin_bytes<0):
        raise ValueError('RAM bound/margin invalid')
    index=directory/'model.safetensors.index.json'
    if not index.is_file() or index.is_symlink():raise ValueError('missing or symlinked shard index')
    sha=hashlib.sha256(index.read_bytes()).hexdigest()
    if expected_index_sha is not None and sha!=expected_index_sha:
        raise ValueError('reference index SHA mismatch')
    obj=json.loads(index.read_text())
    mapping=obj.get('weight_map');claimed=obj.get('metadata',{}).get('total_size')
    if not isinstance(mapping,dict) or not mapping or type(claimed) is not int or claimed<=0:
        raise ValueError('invalid checkpoint index metadata')
    assigned=Counter(mapping.values());shards=sorted(directory.glob('*.safetensors'))
    if (len(shards)!=6 or set(assigned)!={p.name for p in shards} or
            any(p.is_symlink() or not re.fullmatch(r'model-\d{5}-of-00006\.safetensors',p.name)
                for p in shards) or
            {int(p.name[6:11]) for p in shards}!=set(range(1,7))):
        raise ValueError('six unique nonsymlinked checkpoint shards required')
    size=0;seen=set();dtype_counts=Counter()
    for path in shards:
        with safe_open(str(path),framework='pt',device='cpu') as header:
            keys=set(header.keys());expected={key for key,filename in mapping.items() if filename==path.name}
            if keys!=expected or seen&keys:raise ValueError('weight-map/shard key mismatch')
            seen.update(keys)
            for key in keys:
                meta=header.get_slice(key);kind=meta.get_dtype();shape=meta.get_shape()
                if kind not in DTYPE_BYTES or not shape or any(type(x) is not int or x<0 for x in shape):
                    raise ValueError('unsupported or malformed tensor metadata')
                size+=math.prod(shape)*DTYPE_BYTES[kind];dtype_counts[kind]+=1
    if seen!=set(mapping) or size!=claimed:
        raise ValueError('weight_map or declared tensor byte count differs from shard headers')
    max_bytes=max(path.stat().st_size for path in shards)
    return {'schema':'lingbot_legacy_checkpoint_stream_metadata_audit_v1',
            'status':'headers_verified_not_weights_loaded_not_training_approved',
            'index_sha256':sha,'shard_count':len(shards),'tensor_key_count':len(seen),
            'declared_tensor_bytes':claimed,'dtype_counts':dict(sorted(dtype_counts.items())),
            'largest_shard_file_bytes':max_bytes,
            'single_shard_host_lower_bound_bytes':max_bytes+margin_bytes,
            'single_shard_ram_lower_bound_pass':max_bytes+margin_bytes<=available_bytes,
            'memory_available_bytes':available_bytes,'training_approval':False,
            'warning':'Headers only: no tensor bytes hashed/copied, no model key/shape match, no GPU/gradient proof.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--require-reference-index',action='store_true')
    args=p.parse_args()
    from vla_full_load_preflight import available_ram
    result=audit(args.checkpoint_dir,available_bytes=available_ram(),
                 expected_index_sha=REFERENCE_INDEX_SHA if args.require_reference_index else None)
    print(json.dumps(result,sort_keys=True))
    if not result['single_shard_ram_lower_bound_pass']:
        raise SystemExit(4)


if __name__=='__main__':main()

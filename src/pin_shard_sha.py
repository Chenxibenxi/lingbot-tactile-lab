"""Hash six original external safetensors by bounded CPU reads; NO tensor load/GPU.

A manifest produced here identifies bytes at this instant; its publication does
not retroactively certify data provenance or approve tactile-model training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from stream_shard_metadata import audit,REFERENCE_INDEX_SHA
from vla_full_load_preflight import available_ram


def pin(directory,*,available_bytes,expected_index_sha=REFERENCE_INDEX_SHA):
    directory=Path(directory).resolve(strict=True)
    header=audit(directory,available_bytes=available_bytes,expected_index_sha=expected_index_sha)
    manifest={}
    for path in sorted(directory.glob('*.safetensors')):
        before=path.stat();digest=hashlib.sha256()
        with path.open('rb',buffering=1024*1024) as file:
            for block in iter(lambda:file.read(4*1024*1024),b''):
                digest.update(block)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('shard changed during bounded SHA stream')
        manifest[path.name]={'sha256':digest.hexdigest(),'file_bytes':before.st_size}
    if len(manifest)!=6:
        raise ValueError('expected six pinned original shard files')
    return {'schema':'lingbot_external_visual_shard_byte_manifest_v1',
            'checkpoint_index_sha256':header['index_sha256'],
            'shard_files':manifest,'tensor_key_count':header['tensor_key_count'],
            'values_loaded_into_model':False,'training_approval':False,
            'warning':'Read-only whole-file hashes; no model/cuda/gradient/paired-data approval.'}


def main():
    from probe_real_v2_meta import CHECKPOINT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    args=p.parse_args()
    if args.checkpoint_dir.resolve(strict=True)!=CHECKPOINT.resolve(strict=True):
        p.error('only original external 154800 checkpoint may be pinned')
    print(json.dumps(pin(args.checkpoint_dir,available_bytes=available_ram()),sort_keys=True))


if __name__=='__main__':main()

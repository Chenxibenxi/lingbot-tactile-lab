"""CPU-only fail-closed RAM preflight for ORIGINAL LingBotV2 merged FP32 loader.

This does NOT load a checkpoint, inspect private tensor contents, reserve GPU,
approve data, or predict actual peak training memory. The old deploy loader
materializes every safetensors shard in merged_weights BEFORE strict loading;
refuse if even that conservative lower bound plus margin exceeds MemAvailable.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from validate_expert_data import sha256

DEPLOY=Path('/mnt/relaydata/lingbot/lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py')
DEPLOY_SHA='2caf39d560892ccefa0415dd9df3d4c4beab4a0ef4be40a996e5277756b52413'
MARGIN=4*1024**3
SHARDS=6


def inspect_shards(directory, *, available_bytes, margin_bytes=MARGIN):
    directory=Path(directory).resolve(strict=True)
    if type(available_bytes) is not int or available_bytes<=0 or margin_bytes<0:
        raise ValueError('positive host memory bound and nonnegative safety margin required')
    files=sorted(directory.glob('model-*-of-00006.safetensors'))
    if len(files)!=SHARDS or any(p.is_symlink() or not re.fullmatch(r'model-\d{5}-of-00006\.safetensors',p.name)
                                   for p in files) or {int(p.name[6:11]) for p in files}!=set(range(1,7)):
        raise ValueError('expected exactly six original model shards, no missing/symlinked shards')
    sizes=[p.stat().st_size for p in files]
    if any(size<=0 for size in sizes):raise ValueError('empty original model shard')
    total=sum(sizes)
    bound=total+margin_bytes
    return {'schema':'lingbot_original_full_weight_loader_ram_preflight_v1',
            'status':'refuse_full_weight_load' if available_bytes<bound else 'ram_lower_bound_only_not_training_approval',
            'safe_to_attempt_full_loader':available_bytes>=bound,
            'shard_count':len(files),'total_shard_bytes':total,
            'memory_available_bytes':available_bytes,'minimum_required_bytes_with_margin':bound,
            'margin_bytes':margin_bytes,'training_approval':False,
            'note':'Conservative CPU shard lower bound, NOT full VLA/model peak or BF16 gradient memory proof.'}


def available_ram():
    with Path('/proc/meminfo').open() as file:
        for line in file:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable missing; refuse checkpoint loading')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    args=parser.parse_args()
    if not DEPLOY.is_file() or sha256(DEPLOY)!=DEPLOY_SHA:
        parser.error('original legacy server loader changed; read-only RAM estimate invalid')
    loader=DEPLOY.read_text()
    if ('merged_weights[key] = f.get_tensor(key)' not in loader or
            'self.vla.load_state_dict(merged_weights, strict=strict)' not in loader):
        parser.error('cannot establish merged FP32 loader lower bound')
    report=inspect_shards(args.checkpoint_dir,available_bytes=available_ram())
    print(json.dumps(report,sort_keys=True))
    if not report['safe_to_attempt_full_loader']:
        raise SystemExit(4)


if __name__=='__main__':main()

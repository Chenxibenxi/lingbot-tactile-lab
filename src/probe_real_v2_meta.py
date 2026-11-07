"""One-shot REAL patched V2 class meta-device key/shape probe, NO weights/fit.

Requires external GPU lease before import: upstream Triton queries CUDA at module
import even though we only instantiate parameter tensors on the meta device.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

from patch_model_v2 import SOURCE as MODEL_SOURCE,patch_model
from stream_shard_metadata import REFERENCE_INDEX_SHA
from validate_expert_data import sha256
from vla_full_load_preflight import available_ram
from vla_meta_shape_gate import compare_meta_model

MODEL_SHA='9b3c9a2ad6823f9d58da51b732477785a1e4db0d29db80e030e094c7b605e40c'
TRAIN=Path('/home/admin001/PhiZJU/submission/03_模型权重/lingbot-vla-v2-6b-lora-1f-154800/lingbotvla_cli.yaml')
TRAIN_SHA='daa533fe9380a8427fc8737cc5fbbfac8e5192057231e78cd28450319dabbbf0'
TOKENIZER=Path('/mnt/relaydata/lingbot/weights/Qwen3-VL-4B-Instruct')
TOKENIZER_SHA='edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d'
CHECKPOINT=Path('/home/admin001/PhiZJU/submission/03_模型权重/lingbot-vla-v2-6b-lora-1f-154800/checkpoints/global_step_154800/hf_ckpt')


def verify_sources():
    if (sha256(MODEL_SOURCE)!=MODEL_SHA or sha256(TRAIN)!=TRAIN_SHA or
            sha256(TOKENIZER/'config.json')!=TOKENIZER_SHA or
            sha256(CHECKPOINT/'model.safetensors.index.json')!=REFERENCE_INDEX_SHA):
        raise ValueError('original model/train/tokenizer/checkpoint source SHA changed')
    if TRAIN.is_symlink() or TOKENIZER.is_symlink() or CHECKPOINT.is_symlink():
        raise ValueError('external reference source paths must not be symlinks')
    return {'model_source_sha256':MODEL_SHA,'training_config_sha256':TRAIN_SHA,
            'local_tokenizer_config_sha256':TOKENIZER_SHA,'checkpoint_index_sha256':REFERENCE_INDEX_SHA}


def build_old_config():
    import yaml
    from transformers import AutoConfig
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    config_file=yaml.safe_load(TRAIN.read_text())
    values={**config_file['model'],**config_file['train']}
    config=LingbotVLAV2Config(**values)
    for key,value in values.items():
        if not hasattr(config,key):setattr(config,key,value)
    config.attention_implementation='eager'
    config.tokenizer_path=str(TOKENIZER)
    qwen=AutoConfig.from_pretrained(str(TOKENIZER),local_files_only=True)
    qdict=qwen.to_dict();text=qdict.get('text_config',{})
    for key in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
                'num_key_value_heads','rms_norm_eps','rope_theta','vocab_size',
                'max_position_embeddings','hidden_act','tie_word_embeddings','tokenizer_path'):
        if key in text:setattr(config,key,text[key])
        elif key in qdict:setattr(config,key,qdict[key])
    config.vision_config=qwen.vision_config
    if config_file['model'].get('vocab_size',0):config.vocab_size=config_file['model']['vocab_size']
    config.use_cache=True
    config.tactile_enabled=False  # original visual keys ONLY; attach after strict load
    return config


def probe(repo):
    import torch
    repo=Path(repo).resolve(strict=True)
    if repo!=Path(__file__).resolve().parents[1]:raise ValueError('only original relay checkout')
    evidence=verify_sources()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('upstream real-model import requires leased single CUDA device')
    source_text=patch_model(MODEL_SOURCE.read_text())
    preview=repo/'work/real_v2_meta_preview.py'
    if preview.exists():raise FileExistsError('refuse overwrite of private generated model source')
    preview.parent.mkdir(exist_ok=True)
    with preview.open('x',encoding='utf-8') as file:file.write(source_text)
    spec=importlib.util.spec_from_file_location(
        'lingbotvla.models.vla.lingbot_vla._tactile_meta_preview',str(preview))
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config=build_old_config()
    with torch.device('meta'):
        model=module.LingbotVlaV2Policy(config,eval=True)
    audit=compare_meta_model(model,CHECKPOINT,available_bytes=available_ram())
    audit.update(source_evidence_sha256=evidence,
                 isolated_preview_sha256=sha256(preview),
                 cuda_allocated_bytes=int(torch.cuda.memory_allocated()),
                 full_visual_checkpoint_loaded=False,
                 original_model_source_unchanged=sha256(MODEL_SOURCE)==MODEL_SHA)
    if audit['cuda_allocated_bytes']>512*1024**2:
        raise RuntimeError('meta-only probe allocated excessive CUDA memory')
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(probe(args.repo),sort_keys=True))


if __name__=='__main__':main()

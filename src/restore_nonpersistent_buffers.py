"""Restore the audited non-checkpoint buffers after meta/to_empty allocation.

Strict checkpoint keys do not cover persistent=False buffers. Fail closed on
unknown buffers; reconstruct tiny rotary modules using installed constructors,
never instantiate a second full model or alter the checkpoint.
"""
import hashlib
import torch


def restore_nonpersistent_buffers(model):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    rows=[]
    for path, module in model.named_modules():
        names=module._non_persistent_buffers_set
        if not names:
            continue
        reference=None
        with torch.device('cpu'):
            if type(module) is Qwen3VLVisionRotaryEmbedding:
                if names != {'inv_freq'}:raise ValueError('vision rotary buffer contract changed')
                reference=Qwen3VLVisionRotaryEmbedding(module.inv_freq.numel()*2)
            elif type(module) in (Qwen3VLTextRotaryEmbedding,Qwen2RotaryEmbedding):
                if names != {'inv_freq'}:raise ValueError('text rotary buffer contract changed')
                reference=type(module)(config=module.config,device=torch.device('cpu'))
            elif type(module).__name__=='Qwen2TokenMoeBlock':
                if names != {'tokens_per_expert','last_tokens_per_expert','avg_topk_sigmoid_score'}:
                    raise ValueError('MoE counter contract changed')
            elif type(module).__name__=='Qwen2FusedExperts':
                if names != {'_gate_up_proj_cache'} or module._buffers['_gate_up_proj_cache'] is not None:
                    raise ValueError('MoE cache must be empty before first use')
            else:
                raise ValueError(f'unhandled nonpersistent buffers: {path} {type(module).__name__}')
        for name in sorted(names):
            old=module._buffers[name]
            if old is None:
                rows.append({'name':path+'.'+name,'kind':'empty_cache'})
                continue
            if old.is_meta:raise ValueError('allocate model before restoring buffers')
            cpu=(reference._buffers[name] if reference is not None else
                 torch.zeros(old.shape,dtype=old.dtype,device='cpu'))
            if cpu.shape!=old.shape:raise ValueError('reconstructed buffer shape mismatch')
            restored=cpu.to(device=old.device,dtype=old.dtype)
            module._buffers[name]=restored
            if reference is not None and hasattr(module,'original_inv_freq'):
                module.original_inv_freq=restored
                module.attention_scaling=reference.attention_scaling
                module.max_seq_len_cached=reference.max_seq_len_cached
                module.original_max_seq_len=reference.original_max_seq_len
            rows.append({'name':path+'.'+name,'shape':list(restored.shape),'dtype':str(restored.dtype),
                         'sha256':hashlib.sha256(restored.detach().cpu().float().numpy().tobytes()).hexdigest()})
    model._nonpersistent_buffer_restore=rows
    return rows

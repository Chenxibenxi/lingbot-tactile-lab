"""One-shot 154800 BF16 strict visual WEIGHT load (no forward/grad/train).

Caller MUST hold the exclusive relay RTX3090 lease and enforce timeout/Strata
restore. Original weights stay external and read-only; failed process is never
reused. All six files are re-hashed immediately before per-shard value copy.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib import error,request
import torch
from model_attach_contract import attach_visual_identity_adapter
from patch_model_v2 import SOURCE as MODEL_SOURCE,patch_model
from probe_real_v2_meta import CHECKPOINT,build_old_config,verify_sources
from stream_shard_metadata import REFERENCE_INDEX_SHA
from strict_stream_visual_weights import stream_into_allocated_model
from validate_expert_data import sha256
from vla_full_load_preflight import available_ram
from vla_meta_shape_gate import compare_meta_model

PREVIEW_SHA='a80be3e9b7f7834676a086c1ffab964e6a25ad8bc81b4572e8939e0f400b8191'
MARGIN_GPU=6*1024**3
MAX_ALLOCATED=18*1024**3
MAX_INFERENCE_PEAK=22*1024**3
MANIFEST=Path(__file__).resolve().parents[1]/'datasets/relay_154800_shard_sha_v1.json'
LEASE_TOOL=Path('/home/admin001/ai-video-studio/tools/gpu_lease.py')
LEASE_OWNER='lingbot-tactile-lab/real-v2-bf16-stream'


def _lease_parent_pids():
    """Allow `bash --pid $$` -> `timeout` -> this Python process, not unrelated PIDs."""
    pid=os.getpid();parents=set()
    for _ in range(2):
        status=(Path('/proc')/str(pid)/'status').read_text()
        parent=next((int(line.split(':',1)[1].strip()) for line in status.splitlines()
                     if line.startswith('PPid:')),0)
        if parent<=1:break
        parents.add(parent);pid=parent
    return parents


def require_gpu_window():
    """Refuse direct calls, another interactive waiter or a running Strata."""
    if os.uname().nodename!='admin001-MS-7D90':
        raise RuntimeError('only relay single-GPU host may load original VLA values')
    info=json.loads(subprocess.run([sys.executable,str(LEASE_TOOL),'status','--json'],
                    check=True,capture_output=True,text=True,timeout=10).stdout)
    lease=info.get('lease') or {}
    if (info.get('state')!='held' or info.get('holder')!=LEASE_OWNER or
            info.get('pid_alive') is not True or lease.get('pid') not in _lease_parent_pids() or
            info.get('remaining_s',0)<600 or
            any(waiter.get('owner')!=LEASE_OWNER for waiter in info.get('waiting',[]))):
        raise RuntimeError('requires own live ancestor-shell lease TTL>=600 and no other GPU waiter')
    compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid',
                            '--format=csv,noheader'],check=True,capture_output=True,
                           text=True,timeout=10).stdout
    if compute.strip():
        raise RuntimeError('GPU already has compute tasks; do not evict Strata or other owners')
    try:
        with request.urlopen('http://127.0.0.1:8080/health',timeout=3) as response:
            health=json.load(response)
        if not isinstance(health,dict) or health.get('service')!='strata' or health.get('loaded') is not False:
            raise RuntimeError('Strata already loaded or health ambiguous; do not dual allocate GPU')
    except error.URLError as exc:
        if not isinstance(exc.reason,ConnectionRefusedError):
            raise RuntimeError('Strata health unavailable or ambiguous; refuse GPU allocation') from exc
        # A closed local port is acceptable only after exclusive lease/empty GPU.
    return True


def verify_fixed_inputs(repo):
    repo=Path(repo).resolve(strict=True)
    if repo!=Path(__file__).resolve().parents[1]:
        raise ValueError('only original relay tactile checkout')
    anchors=verify_sources()
    preview=repo/'work/real_v2_meta_preview.py'
    if not preview.exists():
        generated=patch_model(MODEL_SOURCE.read_text())
        import hashlib
        if hashlib.sha256(generated.encode()).hexdigest()!=PREVIEW_SHA:
            raise ValueError('generated isolated V2 source SHA mismatch')
        preview.parent.mkdir(exist_ok=True)
        with preview.open('x',encoding='utf-8') as file:file.write(generated)
    if not preview.is_file() or preview.is_symlink() or sha256(preview)!=PREVIEW_SHA:
        raise ValueError('generated isolated V2 source missing or SHA mismatch')
    if not MANIFEST.is_file() or MANIFEST.is_symlink():raise ValueError('external shard SHA card missing')
    manifest=json.loads(MANIFEST.read_text())
    if (manifest.get('schema')!='lingbot_external_visual_shard_byte_manifest_v1' or
            manifest.get('checkpoint_index_sha256')!=REFERENCE_INDEX_SHA or
            manifest.get('tensor_key_count')!=1624 or
            manifest.get('training_approval') is not False or
            len(manifest.get('shard_files',{}))!=6):
        raise ValueError('external shard SHA card not reviewed or mismatched')
    for name,info in manifest['shard_files'].items():
        p=CHECKPOINT/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=info.get('file_bytes'):
            raise ValueError('external original shard missing or file size changed')
    return anchors,preview,manifest


def count_unfrozen_base_parameters(model):
    """Only `model.tactile_adapter.*` may remain trainable on the real wrapper."""
    if not isinstance(getattr(model,'model',None),torch.nn.Module):
        raise ValueError('expected top-level LingbotVlaV2Policy.model wrapper')
    return sum(int(p.requires_grad) for name,p in model.named_parameters()
               if not name.startswith('model.tactile_adapter.'))


def compare_visual_and_zero(model, obs, tactile, config, evidence):
    """Two real sample_actions calls, same TRAIN RGB/state and fresh-cloned noise."""
    core=model.model
    device=core.state_proj.weight.device
    dtype=core.state_proj.weight.dtype
    images=obs['images'].unsqueeze(0).to(device=device,dtype=dtype)
    masks=obs['img_masks'].unsqueeze(0).to(device=device)
    tokens=obs['lang_tokens'].unsqueeze(0).to(device=device)
    token_masks=obs['lang_masks'].unsqueeze(0).to(device=device)
    state=obs['state'].unsqueeze(0).to(device=device,dtype=dtype)
    grid=obs['image_grid_thw'].to(device=device,dtype=torch.long)
    seed_generator=torch.Generator(device=device).manual_seed(97021)
    noise=torch.randn((1,config.n_action_steps,config.max_action_dim),
                      generator=seed_generator,device=device,dtype=dtype)
    model.eval()
    with torch.inference_mode():
        visual=core.sample_actions(images,masks,tokens,token_masks,state,
                                   noise=noise.clone(),image_grid_thw=grid).detach().clone()
        zero=core.sample_actions(images,masks,tokens,token_masks,state,
                noise=noise.clone(),image_grid_thw=grid,
                **{key:value.to(device=device,dtype=dtype if value.is_floating_point() else value.dtype)
                   for key,value in tactile.items()}).detach().clone()
    if visual.shape!=(1,config.n_action_steps,config.max_action_dim) or not torch.isfinite(visual).all():
        raise RuntimeError('real visual action chunk shape or finite check failed')
    peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0
    max_delta=(visual-zero).abs().max().item()
    if not torch.equal(visual,zero) or peak>MAX_INFERENCE_PEAK:
        raise RuntimeError(f'real visual/zero identity or memory failed: max_abs_delta={max_delta}, peak={peak}')
    return {'schema':'lingbot_real_v2_visual_zero_action_identity_v1',
            'status':'actual_qwen_sample_actions_visual_equals_untrained_zero_tactile_only',
            'action_shape':list(visual.shape),'source_evidence':evidence,
            'max_abs_difference':max_delta,'bitwise_equal':True,
            'num_denoise_steps':config.num_steps,'cuda_peak_bytes':peak,
            'training_approval':False,'adapter_backward_run':False,
            'server_rpc_run':False,'official_success_rate_measured':False}


def run(repo,*,visual_zero_smoke=False):
    require_gpu_window()  # before CUDA context or model allocation
    anchors,preview,manifest=verify_fixed_inputs(repo)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('exclusive single NVIDIA GPU required; caller must hold lease')
    from stream_shard_metadata import audit
    memory=available_ram()
    headers=audit(CHECKPOINT,available_bytes=memory,expected_index_sha=REFERENCE_INDEX_SHA)
    if not headers['single_shard_ram_lower_bound_pass'] or headers['dtype_counts']!={'F32':1624}:
        raise ValueError('full original single-shard CPU memory or dtype gate refused')
    gpu_free_before,_=torch.cuda.mem_get_info(0)
    projected=headers['declared_tensor_bytes']//2
    if gpu_free_before < projected+MARGIN_GPU:
        raise RuntimeError('insufficient free GPU for BF16 visual model + bounded 6GiB margin')
    smoke_input=None
    if visual_zero_smoke:
        from prepare_real_vla_visual_smoke import prepare
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
        smoke_input=prepare(repo)
        apply_lingbot_qwen3_vl_patch()  # original visual deploy does this before construction
        config=smoke_input[0]
    else:
        config=build_old_config()
    spec=importlib.util.spec_from_file_location(
        'lingbotvla.models.vla.lingbot_vla._tactile_meta_preview',str(preview))
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with torch.device('meta'):
        model=module.LingbotVlaV2Policy(config,eval=True)
    meta_evidence=compare_meta_model(model,CHECKPOINT,available_bytes=available_ram())
    if meta_evidence['matched_key_count']!=1624 or not meta_evidence['model_is_meta_only']:
        raise AssertionError('real original visual meta model does not match 154800 headers')
    model.to(dtype=torch.bfloat16)  # meta-device dtype change, NO weight bytes yet
    if any(not t.is_meta or t.dtype!=torch.bfloat16 for t in model.state_dict().values()):
        raise RuntimeError('unexpected nonmeta or nonBF16 tensor before bounded CUDA allocation')
    model.to_empty(device=torch.device('cuda:0'))
    allocated_before_load=torch.cuda.memory_allocated()
    if allocated_before_load>MAX_ALLOCATED or torch.cuda.mem_get_info(0)[0]<4*1024**3:
        raise RuntimeError('original BF16 model exceeded allocation or post-allocation GPU headroom')
    proof=stream_into_allocated_model(model,CHECKPOINT,
        available_bytes=available_ram(),expected_index_sha=REFERENCE_INDEX_SHA,
        expected_shard_manifest=manifest)
    torch.cuda.synchronize()
    if proof.copied_key_count!=1624 or proof.source_tensor_bytes!=headers['declared_tensor_bytes']:
        raise AssertionError('strict original visual checkpoint not fully copied')
    attached=attach_visual_identity_adapter(model,proof)
    torch.cuda.synchronize()
    used=torch.cuda.memory_allocated();peak=torch.cuda.max_memory_allocated()
    unfrozen_base=count_unfrozen_base_parameters(model)
    if used>MAX_ALLOCATED or peak>MAX_ALLOCATED or unfrozen_base:
        raise RuntimeError(f'postload BF16 peak/freeze exceeds boundary: '
                           f'used={used} peak={peak} unfrozen_base_count={unfrozen_base}')
    if attached['training_approval'] or attached['server_capability_advertised']:
        raise RuntimeError('zero adapter must not be approved or advertised')
    result={'schema':'lingbot_real_154800_bf16_strict_stream_load_v1',
            'status':'strict_real_visual_weights_loaded_zero_adapter_attached_no_forward_no_training',
            'checkpoint_index_sha256':proof.checkpoint_index_sha256,
            'source_evidence_sha256':anchors,'generated_preview_sha256':PREVIEW_SHA,
            'shard_sha256_match_all_six':True,'keys_loaded':proof.copied_key_count,
            'allocated_before_values_bytes':allocated_before_load,
            'cuda_allocated_after_adapter_bytes':used,'cuda_peak_bytes':peak,
            'model_param_device':'cuda:0','model_param_dtype':'torch.bfloat16',
            'adapter_parameter_count':attached['adapter_parameter_count'],
            'weights_ever_saved_to_repo':False,'model_forward_run':False,
            'adapter_backward_run':False,'training_approval':False,
            'server_capability_advertised':False}
    if visual_zero_smoke:
        _,observation,tactile,evidence=smoke_input
        numerical=compare_visual_and_zero(model,observation,tactile,config,evidence)
        result.update(status='strict_real_visual_load_and_true_zero_tactile_sample_actions_identity_only',
                      model_forward_run=True,sample_actions_run=True,
                      numerical_identity=numerical,cuda_peak_bytes=torch.cuda.max_memory_allocated())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--confirm-visual-only-no-training',action='store_true')
    p.add_argument('--visual-zero-smoke',action='store_true',
                   help='two full real Qwen sample_actions on one fixed TRAIN source, no gradients')
    args=p.parse_args()
    if not args.confirm_visual_only_no_training:p.error('no weights will load without opt-in visual-only flag')
    print(json.dumps(run(args.repo,visual_zero_smoke=args.visual_zero_smoke),sort_keys=True))


if __name__=='__main__':main()

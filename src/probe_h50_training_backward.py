"""Real 154800 H50 flow-loss backward gate. No optimizer or checkpoint writes.

Explicit technical experiment authorized 2026-11-06. Does not promote candidate
seed100023 to approved training/validation, and does not advertise tactile RPC.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import time
import torch
from probe_real_v2_bf16_stream import (require_gpu_window, verify_fixed_inputs,
    count_unfrozen_base_parameters, MAX_INFERENCE_PEAK)
from probe_real_v2_meta import CHECKPOINT, build_old_config
from stream_shard_metadata import REFERENCE_INDEX_SHA
from vla_full_load_preflight import available_ram
from vla_meta_shape_gate import compare_meta_model
from strict_stream_visual_weights import stream_into_allocated_model
from model_attach_contract import attach_visual_identity_adapter
from validate_expert_data import sha256

REPO=Path(__file__).resolve().parents[1]


def validate_batch(batch):
    if batch['actions'].shape != (1,50,55) or batch['joint_mask'].shape != (1,50,55):
        raise ValueError('real H50 padded action/mask shape required')
    if not torch.all(batch['joint_mask'].sum(-1)==14):
        raise ValueError('exactly 14 real action dimensions, never train padded dimensions')
    if batch['action_is_pad'].bool().any():
        raise ValueError('no padded/fabricated horizon labels')
    if any(not torch.isfinite(v).all() for v in batch.values() if torch.is_tensor(v)):
        raise ValueError('nonfinite transformed input')


def prepare(config):
    from types import SimpleNamespace
    import yaml
    from torchvision.transforms.v2 import Resize
    from h50_online_samples import OnlineH50Candidate
    from prepare_real_vla_visual_smoke import TRAIN,TOKENIZER,NORM,ROBOT,CAMERAS
    from lingbotvla.models import build_processor
    from lingbotvla.data.vla_data.utils import FeatureTransform
    reader=OnlineH50Candidate()
    # Fixed first valid anchor, no result-driven frame selection.
    item=reader.sample(0)
    inp=item['input']
    transform=FeatureTransform(str(ROBOT),SimpleNamespace(**yaml.safe_load(TRAIN.read_text())['data']),
        config,build_processor(str(TOKENIZER)),chunk_size=50,norm_stats_path=str(NORM))
    resize=Resize((256,256))
    obs={name:resize(torch.as_tensor(inp['images_rgb'][cam]).permute(2,0,1).contiguous().float())
         for cam,name in CAMERAS.items()}
    obs.update({'observation.state':torch.as_tensor(inp['command_state_14']),
                'task':inp['instruction'],
                'action':torch.as_tensor(item['target']['future_expert_take_action_14']).clone(),
                'action_is_pad':torch.zeros(50,dtype=torch.bool)})
    transformed=transform.apply(obs,policy_eval=False)
    keep=('images','img_masks','state','lang_tokens','lang_masks','actions','joint_mask','action_is_pad','image_grid_thw')
    batch={k:transformed[k] if k=='image_grid_thw' else transformed[k].unsqueeze(0) for k in keep}
    validate_batch(batch)
    tactile={'tactile_history':torch.as_tensor(inp['tactile_history']).unsqueeze(0),
             'tactile_mask':torch.as_tensor(inp['tactile_mask']).unsqueeze(0),
             'tactile_age_ms':torch.tensor([float(inp['tactile_age_ms'])]),
             'tactile_valid':torch.tensor([True])}
    report={'trial_id':item['metadata']['trial_id'],'seed':100023,'sample_index':0,
            'frame_index':item['metadata']['frame_index'],
            'source_card_sha256':sha256(REPO/item['metadata']['source_card']),
            'input_capture_physics_step':inp['capture_physics_step'],
            'action_shape_before_transform':[50,14],
            'positive_force_bins':int((inp['tactile_history'][...,:2]>0).sum()),
            'normalization':'fixed log1p identity-scale diagnostic only; no fitted stats',
            'model_fit_approval':False}
    return batch,tactile,report


def load_model(config,preview,manifest):
    # The optional eval-only atomic MoE kernel fails repeated fixed-input
    # equality. Use the existing group_gemm path in BOTH train and eval.
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert
    qwen2_action_expert.robby_moe_forward=None
    spec=importlib.util.spec_from_file_location('lingbotvla.models.vla.lingbot_vla._h50_backward_preview',str(preview))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with torch.device('meta'):model=module.LingbotVlaV2Policy(config,eval=True)
    gate=compare_meta_model(model,CHECKPOINT,available_bytes=available_ram())
    if gate['matched_key_count']!=1624:raise ValueError('strict meta key mismatch')
    model.to(dtype=torch.bfloat16);model.to_empty(device=torch.device('cuda:0'))
    proof=stream_into_allocated_model(model,CHECKPOINT,available_bytes=available_ram(),
        expected_index_sha=REFERENCE_INDEX_SHA,expected_shard_manifest=manifest)
    from restore_nonpersistent_buffers import restore_nonpersistent_buffers
    restore_nonpersistent_buffers(model)
    attached=attach_visual_identity_adapter(model,proof)
    if count_unfrozen_base_parameters(model):raise ValueError('unfrozen base')
    return model,attached


def run():
    require_gpu_window()
    evidence,preview,manifest=verify_fixed_inputs(REPO)
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    apply_lingbot_qwen3_vl_patch()
    torch.manual_seed(97026)
    cfg=build_old_config();cfg.use_cache=False
    batch,tactile,source=prepare(cfg)
    print('H50_SOURCE_READY',json.dumps(source),flush=True)
    if torch.cuda.mem_get_info()[0]<19*1024**3:raise RuntimeError('insufficient GPU headroom')
    model,attached=load_model(cfg,preview,manifest)
    model.train()
    to_gpu=lambda v:v.to(device='cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
    batch={k:to_gpu(v) for k,v in batch.items()};tactile={k:to_gpu(v) for k,v in tactile.items()}
    noise=torch.randn(batch['actions'].shape,device='cuda',dtype=torch.bfloat16)
    flow_time=torch.tensor([0.5],device='cuda',dtype=torch.bfloat16)
    adapter=model.model.tactile_adapter
    before={k:v.detach().clone() for k,v in adapter.state_dict().items()}
    # Train-mode flow matching identity, fixed noise and t; no inference shortcut.
    with torch.no_grad():
        visual=model(**batch,noise=noise.clone(),time=flow_time)[0].detach()
        zero=model(**batch,noise=noise.clone(),time=flow_time,**tactile)[0].detach()
    if not torch.equal(visual,zero):raise RuntimeError('train flow loss zero identity failed')
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    loss=model(**batch,noise=noise.clone(),time=flow_time,**tactile)[0]
    if not loss.requires_grad or not torch.isfinite(loss):raise RuntimeError('flow loss lacks finite gradient path')
    loss.backward();torch.cuda.synchronize()
    grads={name:{'present':p.grad is not None,
                  'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),
                  'l2':None if p.grad is None else float(p.grad.float().norm())}
           for name,p in adapter.named_parameters()}
    base_grad=[n for n,p in model.named_parameters() if not n.startswith('model.tactile_adapter.') and p.grad is not None]
    unchanged=all(torch.equal(v,before[k]) for k,v in adapter.state_dict().items())
    peak=torch.cuda.max_memory_allocated()
    if base_grad or not unchanged or peak>MAX_INFERENCE_PEAK or not all(v['finite'] for v in grads.values()):
        raise RuntimeError('gradient ownership, finite, unchanged-weight or memory gate failed')
    if grads['to_state.weight']['l2']<=0:raise RuntimeError('adapter output projection has no learning signal')
    return {'schema':'lingbot_real_h50_backward_gate_v1','status':'passed_technical_backward_only',
        'source':source,'source_evidence':evidence,'keys_loaded':1624,
        'all_six_original_shard_hashes_verified':True,'adapter_parameters':attached['adapter_parameter_count'],
        'loss':float(loss.detach()),'visual_loss':float(visual),'zero_loss_identity':True,
        'adapter_gradients':grads,'base_parameters_with_grad':base_grad,
        'adapter_weights_unchanged':unchanged,'optimizer_steps':0,
        'cuda_peak_allocated_bytes':peak,'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'forward_backward_seconds':time.monotonic()-start,'training_dataset_approved':False,
        'normalization_approved':False,'tactile_improvement_claim':False,'rpc_enabled':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve()
    if not out.is_relative_to(REPO/'data/reports') or out.exists():raise ValueError('fresh private report path required')
    out.parent.mkdir(parents=True,exist_ok=True)
    try:result=run()
    except Exception as exc:
        with out.open('x') as f:json.dump({'status':'failed','error_type':type(exc).__name__,'error':str(exc),'optimizer_steps':0},f,indent=2)
        raise
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

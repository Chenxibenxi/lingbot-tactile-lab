"""Finite, leased GPU performance experiment; no research checkpoint is written."""
import argparse
import json
import time
from pathlib import Path
import torch
import run_cwmj_m7_training as base
from microbatch_collator import collate
from microbatch_comparison import tensor_digest
from microbatch_loss_weighting import weighted_micro_loss

R=Path(__file__).resolve().parents[1]


def run(micro, out):
    from verify_microbatch_benchmark_v2 import verify
    manifest_sha=verify()
    from verify_m7_training import verify as verify_training
    verify_training(R)
    base.require_reproducibility_gate()
    base.require_gpu_window()
    plan_path=R/'datasets/microbatch_throughput_windows_v1.json'
    plan=json.loads(plan_path.read_text())
    inventory=R/'data/reports/microbatch_real_cpu_inventory_v1.json'
    if base.sha256(inventory)!=plan['inventory_sha256']:
        raise ValueError('CPU inventory drift')
    evidence,preview,weights=base.verify_fixed_inputs(R)
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    apply_lingbot_qwen3_vl_patch()
    torch.manual_seed(97040)
    cfg=base.build_old_config();cfg.use_cache=False;cfg.num_steps=10;cfg.precompute_grid_thw=False
    if cfg.align_params!={} or getattr(cfg,'sequence_wise_mode',None)!='per_sequence':
        raise ValueError('unsupported auxiliary objective')
    started=time.monotonic()
    items,stats,sources=base.transform_samples(cfg)
    train=[x for x in items if x['meta']['fold']=='train']
    if base.dataset_fingerprint(items)!=plan['transformed_dataset_sha256']:
        raise ValueError('transformed inputs drift')
    selected=[train[i] for i in plan['indices']]
    if len(selected)!=32 or [x['meta'] for x in selected]!=plan['samples']:
        raise ValueError('fixed sample selection drift')
    prepare_seconds=time.monotonic()-started
    started=time.monotonic()
    model,_=base.load_model(cfg,preview,weights)
    if model.model.qwenvl_with_expert.config.precompute_grid_thw is not False:raise ValueError("visual grid cache must be disabled")
    adapter=model.model.tactile_adapter;adapter.float();adapter.set_normalization(stats)
    initialization=base.reinitialize_adapter(adapter,98101)
    load_seconds=time.monotonic()-started
    model.eval();started=time.monotonic()
    for index,item in zip(plan['indices'],selected):
        item['teacher']=base.infer(model,item['batch'],{},71000+index).detach().cpu()
    torch.cuda.synchronize();teacher_seconds=time.monotonic()-started
    generator=torch.Generator(device='cpu').manual_seed(990010)
    noise=torch.randn((32,50,55),generator=generator).to(torch.bfloat16)
    times=torch.linspace(.2,.8,32).to(torch.bfloat16)
    model.train()
    # Observe actual routing shapes without changing the model's loss calculation.
    original=model.model._moe_losses_and_metrics
    router_tokens=set()
    def observed(logits,losses):
        if not logits:raise ValueError('expected configured MoE router outputs')
        b=losses.shape[0]
        counts={int(x.shape[0])//b for x in logits if x is not None}
        if len(counts)!=1 or any(x.shape[0]%b for x in logits if x is not None):
            raise ValueError('unequal router token contract')
        router_tokens.update(counts)
        if len(router_tokens)!=1:raise ValueError('router tokens/sample changed')
        return original(logits,losses)
    model.model._moe_losses_and_metrics=observed

    def optimizer():return torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01)
    def step(indices,size,opt,update=True):
        opt.zero_grad(set_to_none=True)
        denominator=sum(int((selected[i]['batch']['joint_mask'].bool() & selected[i]['valid'].unsqueeze(-1)).sum()) for i in indices)
        value=0.
        for offset in range(0,len(indices),size):
            ids=indices[offset:offset+size]
            part=collate([selected[i] for i in ids])
            b={k:base.gpu(v) for k,v in part['batch'].items()}
            teacher=base.gpu(torch.cat([selected[i]['teacher'] for i in ids]))
            b['actions'],b['joint_mask']=base.prepare_censored_flow(b['actions'],b['joint_mask'],base.gpu(part['valid']),teacher)
            outputs=model(**b,**{k:base.gpu(v) for k,v in part['tactile'].items()},
                          noise=base.gpu(noise[ids]),time=base.gpu(times[ids]))
            loss=weighted_micro_loss(outputs[0],outputs[1],valid_coordinates=int(b['joint_mask'].sum()),
                global_valid_coordinates=denominator,samples=len(ids),global_samples=len(indices),
                alignment_disabled=True,sequence_mode=cfg.sequence_wise_mode,equal_router_tokens=len(router_tokens)==1)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();value+=float(loss.detach())
            del outputs,loss,b,teacher,part
        if base.count_unfrozen_base_parameters(model) or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('model.tactile_adapter.')):
            raise RuntimeError('base gradient contamination')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise RuntimeError('missing/nonfinite adapter gradient')
        if torch.cuda.max_memory_allocated()>22*1024**3:raise RuntimeError('22 GiB allocation budget exceeded')
        if update:
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient norm')
            opt.step()
        return value,denominator

    # Common micro1 conditioning step makes the initially zero output projection nonzero.
    common=optimizer();step(list(range(4)),1,common)
    snapshot={k:v.detach().clone() for k,v in adapter.state_dict().items()}
    del common
    comparison_identity=dict(conditioned_adapter_sha256=tensor_digest(snapshot),
        teacher_sha256=tensor_digest({str(i):x['teacher'] for i,x in enumerate(selected)}),
        flow_sha256=tensor_digest({'noise':noise,'time':times}),
        plan_sha256=base.sha256(plan_path),
        numeric_screen_sha256=base.sha256(R/'datasets/microbatch_numeric_screen_v1.json'))
    # An un-clipped gradient comparison at the exact same conditioned parameters.
    opt=optimizer();loss,_=step(list(range(4)),micro,opt,update=False)
    torch.save({k:p.grad.detach().cpu() for k,p in adapter.named_parameters()},out/'comparison_gradients.pt')
    (out/'comparison.json').write_text(json.dumps(dict(loss=loss,router_tokens_per_sample=sorted(router_tokens),identity=comparison_identity)))
    # Warm up this micro shape, then restore identical state and a fresh optimizer.
    step(list(range(4)),micro,opt)
    adapter.load_state_dict(snapshot);del opt,snapshot
    opt=optimizer();opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    rows=[]
    for start in range(0,32,4):
        step_started=time.monotonic();value,count=step(list(range(start,start+4)),micro,opt)
        torch.cuda.synchronize()
        rows.append(dict(optimizer_step=start//4+1,loss=value,valid_coordinates=count,seconds=time.monotonic()-step_started))
    elapsed=time.monotonic()-started
    return dict(status='completed',precompute_grid_thw=False,manifest_sha256=manifest_sha,microbatch=micro,global_batch=4,accumulation=4//micro,
        optimizer_steps=8,sample_exposures=32,seconds=elapsed,samples_per_second=32/elapsed,
        valid_coordinates_per_second=sum(x['valid_coordinates'] for x in rows)/elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_free_total_bytes=list(torch.cuda.mem_get_info()),prepare_seconds=prepare_seconds,
        model_load_seconds=load_seconds,teacher_seconds=teacher_seconds,rows=rows,
        initialization=initialization,source_evidence=evidence,plan_sha256=base.sha256(plan_path),
        code_sha256={p:base.sha256(R/p) for p in ('src/run_microbatch_benchmark_v2.py','src/microbatch_collator.py','src/microbatch_loss_weighting.py','src/microbatch_comparison.py')},
        training_approval=False,tactile_benefit_claim=False,comparison_pass=None,
        limitation='Synthetic fixed flow noise/time; throughput experiment, not task-benefit training.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--micro',type=int,choices=(1,2,4),required=True)
    args=parser.parse_args();out=R/f'data/reports/microbatch_benchmark_v2/micro{args.micro}'
    out.mkdir(parents=True,exist_ok=False)
    try:result=run(args.micro,out)
    except Exception as exc:
        result=dict(status='failed',microbatch=args.micro,error=type(exc).__name__+': '+str(exc),training_approval=False)
        (out/'result.json').write_text(json.dumps(result,indent=2));raise
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()

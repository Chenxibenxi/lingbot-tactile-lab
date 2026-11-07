"""Fixed-data duration experiment; requires a separately frozen M8 manifest."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from safetensors.torch import save_file, load_file
import run_cwmj_m7_training as base
from multiseed_tactile_controls import payload_for as detailed_payload
from contact_state_control import contact_state_payload
from duration_training_contract import isolated_validation

ARMS = ('zero', 'true', 'mismatch', 'contact_state')
NODES = (720, 1440, 2880)
R = base.REPO


def payload(item, arm):
    return contact_state_payload(item['tactile']) if arm == 'contact_state' else detailed_payload(item, arm)


def evaluate(model, items, arm):
    with isolated_validation(model):
        return [base.metrics(base.infer(model, x['batch'], {} if arm == 'frozen' else payload(x, arm), 81000+i), x)
                for i, x in enumerate(items)]


def run(out, repeat, arm):
    from verify_m7_training import verify
    verify(R)
    manifest_path = R/'data/reports/cwmj_m8/training_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'cwmj_m8_duration_training_v1' or manifest.get('frozen') is not True:
        raise ValueError('frozen M8 manifest required')
    for path, digest in manifest['files'].items():
        if base.sha256(R/path) != digest:
            raise ValueError(f'M8 input changed: {path}')
    base.require_reproducibility_gate(); base.require_gpu_window()
    evidence, preview, weights = base.verify_fixed_inputs(R)
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    apply_lingbot_qwen3_vl_patch(); torch.manual_seed(97040)
    cfg = base.build_old_config(); cfg.use_cache = False; cfg.num_steps = 10
    items, stats, sources = base.transform_samples(cfg)
    train = [x for x in items if x['meta']['fold'] == 'train']
    val = [x for x in items if x['meta']['fold'] == 'validation']
    historical = R/f'data/reports/cwmj_m{7 if arm == "contact_state" else 6}_training_r{repeat}'
    old = json.loads((historical/'protocol.json').read_text())
    fingerprint = base.dataset_fingerprint(items)
    if fingerprint != old['transformed_dataset_sha256']:
        raise ValueError('transformed dataset changed')
    model, _ = base.load_model(cfg, preview, weights)
    adapter = model.model.tactile_adapter
    adapter.float(); adapter.set_normalization(stats)
    initialization = base.reinitialize_adapter(adapter, 98101+repeat)
    if initialization != old['initialization']:
        raise ValueError('initialization changed')
    initial = copy.deepcopy(adapter.state_dict())
    frozen = evaluate(model, val, 'frozen')
    if frozen != json.loads((historical/'frozen_validation.json').read_text()):
        raise ValueError('frozen validation changed')
    protocol = dict(repeat=repeat, arm=arm, nodes=NODES, sources=sources,
                    dataset_sha256=fingerprint, initialization=initialization,
                    manifest_sha256=base.sha256(manifest_path), normalization=stats)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out/'frozen_validation.json').write_text(json.dumps(frozen, indent=2))
    model.eval()  # Match historical teacher generation; no training-mode dropout.
    for i, item in enumerate(train):
        item['teacher'] = base.infer(model, item['batch'], {}, 71000+i).detach().cpu()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=.01)
    order = base.balanced_order(train, NODES[-1], 98201+repeat)
    old_logs = [x for x in json.loads((historical/'training_log.json').read_text()) if x['arm'] == arm]
    if [(train[i]['meta']['seed'], train[i]['meta']['frame_index']) for i in order[:720]] != [(x['seed'], x['frame']) for x in old_logs]:
        raise ValueError('historical sampling prefix changed')
    model.train(); logs=[]; nodes=[]; started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step, index in enumerate(order):
        item=train[index]; batch={k:base.gpu(v) for k,v in item['batch'].items()}
        batch['actions'], batch['joint_mask'] = base.prepare_censored_flow(batch['actions'], batch['joint_mask'], base.gpu(item['valid']), base.gpu(item['teacher']))
        torch.manual_seed(990000+10000*repeat+step); optimizer.zero_grad(set_to_none=True)
        loss=model(**batch, **{k:base.gpu(v) for k,v in payload(item,arm).items()})[0]
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        loss.backward()
        if base.count_unfrozen_base_parameters(model) or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('model.tactile_adapter.')):
            raise RuntimeError('base gradient contamination')
        grad=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
        if not torch.isfinite(grad) or torch.cuda.max_memory_allocated()>22*1024**3:
            raise RuntimeError('gradient or memory budget failed')
        optimizer.step()
        logs.append(dict(step=step+1, loss=float(loss.detach()), grad_norm=float(grad), seed=item['meta']['seed'], frame=item['meta']['frame_index']))
        if (step+1) not in NODES: continue
        node=out/f'step_{step+1}'; node.mkdir()
        model_dir=R/'models'/out.name/f'step_{step+1}'; model_dir.mkdir(parents=True,exist_ok=False)
        artifact=model_dir/'adapter.safetensors'
        state={k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()}
        save_file(state,str(artifact)); torch.save(optimizer.state_dict(),model_dir/'optimizer.pt')
        if step+1 == 720:
            previous=load_file(str(R/'models'/historical.name/arm/'adapter.safetensors'))
            if state.keys()!=previous.keys() or any(not torch.equal(state[k],previous[k]) for k in state):
                raise RuntimeError('720-step weights differ from historical control')
        metrics=evaluate(model,val,arm)
        if step+1==720 and metrics!=json.loads((historical/'result.json').read_text())['results'][arm]:
            raise RuntimeError('720-step validation differs from historical control')
        result=dict(status='completed',steps=step+1,arm=arm,repeat=repeat,validation=metrics,frozen=frozen,
                    artifact=str(artifact),sha256=base.sha256(artifact),
                    unique_windows=len({(x['seed'],x['frame']) for x in logs}),train_windows=len(train),
                    elapsed_seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated(),
                    historical_720_exact=True,changed_tensors=sum(not torch.equal(v,initial[k]) for k,v in adapter.state_dict().items()))
        (node/'training_log.json').write_text(json.dumps(logs,indent=2))
        (node/'result.json').write_text(json.dumps(result,indent=2))
        from report_cwmj_m8_training import report_node
        report_node(node)
        nodes.append(result)
        print('NODE_COMPLETE',arm,repeat,step+1,flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='completed',nodes=nodes,protocol=protocol),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,choices=range(3),required=True);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args()
    out=R/f'data/reports/cwmj_m8_training_r{a.repeat}_{a.arm}';out.mkdir(parents=True,exist_ok=False)
    try:run(out,a.repeat,a.arm)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc)),indent=2));raise

"""M7 contact-state-only extension; M6 sources and base remain read-only."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from terminal_h50_samples import TerminalEpisode
from cwmj_m4_samples import M4TerminalEpisode
from cwmj_m6_samples import M6TerminalEpisode
from training_repeat_contract import reinitialize_adapter,balanced_order,dataset_fingerprint
from multiseed_tactile_controls import assign_mismatches
from contact_state_control import contact_state_payload

def payload_for(item,arm):
    if arm != "contact_state":raise ValueError("M7 contact-only training arm required")
    return contact_state_payload(item["tactile"])
from terminal_h50_contract import prepare_censored_flow,tactile_arm
from probe_h50_training_backward import load_model,REPO
from probe_real_v2_bf16_stream import require_gpu_window,verify_fixed_inputs,count_unfrozen_base_parameters
from probe_real_v2_meta import build_old_config
from contact_normalization import fit_positive_contact_stats
from validate_expert_data import sha256


def select_eval(samples):
    ids=[]
    for contact in (False,True):
        group=[i for i,s in enumerate(samples) if s['metadata']['contact']==contact]
        if len(group)<4:raise ValueError('insufficient preregistered contact stratum')
        ids += [group[i] for i in np.linspace(0,len(group)-1,4,dtype=int)]
    return sorted(set(ids))


def transform_samples(config):
    from types import SimpleNamespace
    import yaml
    from torchvision.transforms.v2 import Resize
    from prepare_real_vla_visual_smoke import TRAIN,TOKENIZER,NORM,ROBOT,CAMERAS
    from lingbotvla.models import build_processor
    from lingbotvla.data.vla_data.utils import FeatureTransform
    transform=FeatureTransform(str(ROBOT),SimpleNamespace(**yaml.safe_load(TRAIN.read_text())['data']),
        config,build_processor(str(TOKENIZER)),chunk_size=50,norm_stats_path=str(NORM))
    resize=Resize((256,256));dataset=[];raw_train=[];sources=[]
    manifest=json.loads((REPO/'data/reports/cwmj_m6/training_manifest.json').read_text())
    from prepare_cwmj_m6_training_manifest import validate_split
    validate_split(manifest['train_seeds'],manifest['validation_seeds'],manifest['new_train_seeds'],manifest['new_validation_seeds'])
    if manifest.get('training_approval') is not True:raise ValueError('fixed research training manifest required')
    if len(manifest['new_train_seeds'])<4 or len(manifest['new_validation_seeds'])<2:raise ValueError('M6 data gate not met')
    if manifest['training_protocol_sha256']!=sha256(REPO/'docs/CWMJ_M6_TRAINING_PROTOCOL_ZH.md'):raise ValueError('frozen training protocol changed')
    for seed in manifest['train_seeds']+manifest['validation_seeds']:
        ep=(M4TerminalEpisode(seed) if seed in (100045,100046,100049) else M6TerminalEpisode(seed))
        expected=next(x['source_sha256'] for x in manifest['sources'] if x['seed']==seed)
        if ep.hashes!=expected:raise ValueError('frozen M6 source bytes changed')
        all_samples=[ep.sample(i,False) for i in range(len(ep))]
        sources.append({'seed':seed,'fold':ep.fold,'source_sha256':ep.hashes,'samples':len(ep),
                        'contact_samples':sum(s['metadata']['contact'] for s in all_samples)})
        if ep.fold=='train':raw_train.extend(all_samples);ids=range(len(ep))
        else:ids=select_eval(all_samples)
        for i in ids:
            s=ep.sample(int(i),True);x=s['input'];y=s['target']
            obs={name:resize(torch.as_tensor(x['images_rgb'][cam]).permute(2,0,1).contiguous().float()) for cam,name in CAMERAS.items()}
            obs.update({'observation.state':torch.as_tensor(x['command_state_14']), 'task':x['instruction'],
                        'action':torch.as_tensor(y['actions_14']).clone(), 'action_is_pad':torch.as_tensor(~y['valid_steps'])})
            out=transform.apply(obs,policy_eval=False)
            batch={k:out[k] if k=='image_grid_thw' else out[k].unsqueeze(0) for k in
                   ('images','img_masks','state','lang_tokens','lang_masks','actions','joint_mask','action_is_pad','image_grid_thw')}
            tactile={'tactile_history':torch.as_tensor(x['tactile_history']).unsqueeze(0),
                'tactile_mask':torch.as_tensor(x['tactile_mask']).unsqueeze(0),
                'tactile_age_ms':torch.tensor([x['tactile_age_ms']]),'tactile_valid':torch.tensor([True])}
            dataset.append({'batch':batch,'tactile':tactile,'valid':torch.as_tensor(y['valid_steps']).unsqueeze(0),
                            'meta':s['metadata'],'real_steps':y['real_steps']})
    stats=fit_positive_contact_stats(raw_train,sorted({x['metadata']['trial_id'] for x in raw_train}))
    stats.update(training_approval=True,approval_scope='20261106_cwmj_m6',
                 authorization='user explicitly authorized GPU training and simulator verification in this task')
    return assign_mismatches(dataset),stats,sources


def gpu(t):return t.to('cuda',dtype=torch.bfloat16 if t.is_floating_point() else t.dtype)

def infer(model,batch,payload,noise_seed):
    kwargs={k:gpu(batch[k]) for k in ('images','img_masks','lang_tokens','lang_masks','state','image_grid_thw')}
    torch.manual_seed(noise_seed)
    noise=torch.randn((1,50,55),device='cuda',dtype=torch.bfloat16)
    # sample_actions requires KV caching even though training forward does not.
    model.config.use_cache=True
    with torch.no_grad():result=model.sample_actions(**kwargs,noise=noise,**{k:gpu(v) for k,v in payload.items()})
    model.config.use_cache=False
    if not torch.isfinite(result).all():raise RuntimeError('nonfinite real sampled actions')
    return result


def metrics(pred,item):
    y=gpu(item['batch']['actions']);j=gpu(item['batch']['joint_mask']).bool()
    valid=gpu(item['valid']);mask=j & valid.unsqueeze(-1)
    err=(pred.float()-y.float()).abs()
    first=mask.clone();first[:,10:]=False
    return {'seed':item['meta']['seed'],'frame':item['meta']['frame_index'],'contact':item['meta']['contact'],
            'real_steps':item['real_steps'],'mae_first10':float(err[first].mean()),'mae_all_valid':float(err[mask].mean())}


def evaluate(model,items,payload_arm):
    model.eval();out=[]
    for idx,item in enumerate(items):
        payload={} if payload_arm=='frozen' else payload_for(item,payload_arm)
        pred=infer(model,item['batch'],payload,81000+idx)
        out.append(metrics(pred,item))
    return out


def require_reproducibility_gate():
    gate=json.loads((REPO/'data/reports/model_reproducibility_gate_v1.json').read_text())
    if not gate['exact_equal'] or not gate['within_process_equal'] or not gate['inputs_equal']:
        raise ValueError('independent model reproducibility gate failed')
    for key,name in [('loader_sha256','probe_h50_training_backward.py'),('restore_sha256','restore_nonpersistent_buffers.py')]:
        if gate[key]!=sha256(REPO/'src'/name):raise ValueError('reproducibility loader code changed')


def run(out,repeat):
    if repeat not in (0,1,2):raise ValueError('three fixed repetitions only')
    from verify_m7_training import verify
    verify(REPO)
    require_reproducibility_gate()
    require_gpu_window();evidence,preview,manifest=verify_fixed_inputs(REPO)
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    apply_lingbot_qwen3_vl_patch();torch.manual_seed(97040)
    cfg=build_old_config();cfg.use_cache=False;cfg.num_steps=10
    items,stats,sources=transform_samples(cfg)
    train=[s for s in items if s['meta']['fold']=='train'];val=[s for s in items if s['meta']['fold']=='validation']
    protocol={'schema':'lingbot_cwmj_m7_contact_state_training','authorization':'user allowed GPU training and local simulation',
              'source_evidence':evidence,'sources':sources,'normalization':stats,'train_samples':len(train),'val_samples':len(val),
              'steps':720,'lr':1e-4,'arms':['contact_state'],'repeat_index':repeat,'initialization_seed':98101+repeat,'order_seed':98201+repeat,'flow_noise_base':990000+10000*repeat,'num_denoise_steps':10,
              'code_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(REPO/'docs/CWMJ_M7_TRAINING_PROTOCOL_ZH.md')}
    protocol['transformed_dataset_sha256']=dataset_fingerprint(items)
    protocol['loader_sha256']=sha256(REPO/'src/probe_h50_training_backward.py')
    protocol['training_manifest_sha256']=sha256(REPO/'data/reports/cwmj_m6/training_manifest.json')
    protocol['repeat_contract_sha256']=sha256(REPO/'src/training_repeat_contract.py')
    protocol['controls_sha256']=sha256(REPO/'src/contact_state_control.py')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('DATA_READY',len(train),len(val),flush=True)
    model,attached=load_model(cfg,preview,manifest)
    (out/'restored_buffers.json').write_text(json.dumps(model._nonpersistent_buffer_restore,indent=2))
    model.model.tactile_adapter.float();model.model.tactile_adapter.set_normalization(stats)
    protocol['initialization']=reinitialize_adapter(model.model.tactile_adapter,98101+repeat)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    old=json.loads((REPO/f'data/reports/cwmj_m6_training_r{repeat}/protocol.json').read_text())
    if protocol['transformed_dataset_sha256']!=old['transformed_dataset_sha256'] or protocol['initialization']!=old['initialization']:raise ValueError('M6 matching gate failed')
    initial=copy.deepcopy(model.model.tactile_adapter.state_dict())
    baseline=evaluate(model,val,'frozen')
    (out/'frozen_validation.json').write_text(json.dumps(baseline,indent=2))
    # Teacher completions generated once for training only, no validation targets.
    for i,item in enumerate(train):
        pred=infer(model,item['batch'],{},71000+i).detach().cpu()
        item['teacher']=pred
        if i%10==0:print('TEACHER',i,len(train),flush=True)
    # Real GPU exact invariance: changing all missing sentinels leaves model input identical.
    first=train[-1];b={k:gpu(v) for k,v in first['batch'].items()};v=gpu(first['valid']);t=gpu(first['teacher'])
    filled,mask=prepare_censored_flow(b['actions'],b['joint_mask'],v,t)
    sent=b['actions'].clone();sent[~v.unsqueeze(-1).expand_as(sent)]=float('nan')
    other,mask2=prepare_censored_flow(sent,b['joint_mask'],v,t)
    if not torch.equal(filled,other) or not torch.equal(mask,mask2):raise RuntimeError('tail sentinel leak')
    zero_payload={k:gpu(z) for k,z in first['tactile'].items()}
    b['actions']=filled;b['joint_mask']=mask
    fixed_noise=torch.randn_like(filled);tm=torch.tensor([.5],device='cuda',dtype=torch.bfloat16)
    with torch.no_grad():
        visual=model(**b,noise=fixed_noise,time=tm)[0]
        zero=model(**b,noise=fixed_noise,time=tm,**zero_payload)[0]
    if not torch.equal(visual,zero):raise RuntimeError('zero adapter identity failed')
    results={'frozen':baseline};updates={};all_logs=[]
    for arm in ('contact_state',):
        adapter=model.model.tactile_adapter;adapter.load_state_dict(initial)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01)
        model.train();torch.cuda.reset_peak_memory_stats();started=time.monotonic();logs=[]
        order=balanced_order(train,720,98201+repeat)
        for step,idx in enumerate(order):
            item=train[int(idx)];batch={k:gpu(v) for k,v in item['batch'].items()}
            batch['actions'],batch['joint_mask']=prepare_censored_flow(batch['actions'],batch['joint_mask'],gpu(item['valid']),gpu(item['teacher']))
            payload={k:gpu(v) for k,v in payload_for(item,arm).items()}
            torch.manual_seed(990000+10000*repeat+step);optimizer.zero_grad(set_to_none=True)
            loss=model(**batch,**payload)[0]
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward()
            if count_unfrozen_base_parameters(model) or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('model.tactile_adapter.')):
                raise RuntimeError('base gradient contamination')
            grad=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('nonfinite adapter grad')
            if torch.cuda.max_memory_allocated()>22*1024**3:raise RuntimeError('memory budget exceeded')
            optimizer.step()
            logs.append({'arm':arm,'step':step+1,'loss':float(loss.detach()),'grad_norm':float(grad),
                         'seed':item['meta']['seed'],'frame':item['meta']['frame_index']})
            if step%10==0:print('TRAIN',arm,step+1,logs[-1]['loss'],flush=True)
        from safetensors.torch import save_file
        model_dir=REPO/'models'/out.name/arm;model_dir.mkdir(parents=True,exist_ok=False)
        artifact=model_dir/'adapter.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},str(artifact))
        torch.save(optimizer.state_dict(),model_dir/'optimizer.pt')
        changed=sum(not torch.equal(v,initial[k]) for k,v in adapter.state_dict().items())
        updates[arm]={'artifact':str(artifact),'sha256':sha256(artifact),'steps':720,'changed_tensors':changed,
                      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'train_seconds':time.monotonic()-started}
        if not changed:raise RuntimeError('optimizer did not change adapter')
        (model_dir/'metadata.json').write_text(json.dumps({'protocol':protocol,'training':updates[arm]},indent=2))
        results[arm]=evaluate(model,val,arm)
        if arm=='true':
            results['true_input_zero']=evaluate(model,val,'zero')
            results['true_input_mismatch']=evaluate(model,val,'mismatch')
        all_logs.extend(logs)
        (out/'progress.json').write_text(json.dumps({'results':results,'updates':updates},indent=2))
        (out/'training_log.json').write_text(json.dumps(all_logs,indent=2))
        from report_cwmj_m7_training import report_arm
        report_arm(out,arm,results,updates,logs,protocol)
        print('ARM_DONE',arm,json.dumps(updates[arm]),flush=True)
        del optimizer
    validation_seeds=sorted({x['meta']['seed'] for x in val})
    (out/'donor_mapping.json').write_text(json.dumps([{'current':x['meta'],'donor':x['mismatch_meta']} for x in items],indent=2))
    summary={}
    for arm,rows in results.items():
        summary[arm]={str(seed):{str(contact):float(np.mean([r['mae_first10'] for r in rows if r['seed']==seed and r['contact']==contact]))
              for contact in (False,True)} for seed in validation_seeds}
    new_validation_seeds=json.loads((REPO/'data/reports/cwmj_m6/training_manifest.json').read_text())['new_validation_seeds']
    support=None  # descriptive offline evaluation; no selection or closed-loop inference
    temporal=None  # no time-mean arm in this replication batch
    report={'status':'completed_real_adapter_training_offline_pilot','results':results,'summary_first10_mae':summary,
            'updates':updates,'predeclared_offline_support':support,'closed_loop_improvement_proven':False,
            'temporal_support':temporal,'independent_train_seeds':len({x['meta']['seed'] for x in train}),'independent_validation_seeds':len(validation_seeds),'rpc_enabled':False,'repeat_index':repeat,'new_validation_seeds':new_validation_seeds}
    (out/'result.json').write_text(json.dumps(report,indent=2));print('PILOT_RESULT',json.dumps(summary),support,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--repeat',type=int,choices=(0,1,2),required=True);args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(REPO/'data/reports') or out.exists():raise ValueError('fresh private run required')
    out.mkdir(parents=True)
    try:run(out,args.repeat)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc)},indent=2));raise

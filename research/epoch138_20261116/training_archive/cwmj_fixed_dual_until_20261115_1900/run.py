"""Fresh-base single-task SFT. Ten real epochs; fail closed on data admission."""
from pathlib import Path
import json,hashlib,random,math,time,os,subprocess,sys,shutil
import torch
import run_cwmj_m7_training as b
from cwmj_full_trajectory_data import FullTrajectoryDataset
from cwmj_model_bridge_v1 import build_feature_transform
from cwmj_temporal_conditioner import CWMJTemporalConditioner
from action_step_conditioner import ActionStepConditioner
from contact_action_projection_lora import ActionProjectionLoRA
from contact_action_film import ContactActionFiLM
R=b.REPO;C=R/'datasets/cwmj_fixed_dual_until_20261115_1900.json';cfg_run=json.loads(C.read_text());D=R/'data/cwmj_fixed_dual_until_20261115_1900';O=R/'data/reports/cwmj_fixed_dual_until_20261115_1900'
PREFIXES=('model.tactile_adapter.','model.action_step_conditioner.','model.action_film.','model.action_out_proj.lora_')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def save(p,v):
 t=p.with_suffix('.tmp');torch.save(v,t);t.replace(p)
admission=json.loads((R/cfg_run['dataset_admission']).read_text());index=R/cfg_run['dataset']
assert admission['training_admitted'] is True and admission['index_sha256']==sha(index)
assert admission['successful_videos']==50 and admission['left_videos']==25 and admission['right_videos']==25
assert admission['human_authorized']
assert admission['training_scope']=='basic_single_task_position_generalization_action_sft' and admission['basic_action_sft_scope_confirmed']
assert cfg_run['launch_allowed'] and cfg_run['training_scope']==admission['training_scope']
assert not O.exists(),'refuse duplicate training launch'
O.mkdir();D.mkdir(exist_ok=True);start=time.monotonic();step=0;epoch=0;params={};optimizer=None;last_guard=0
status={'status':'loading','initialization':cfg_run['initialization'],'deadline_local':cfg_run['deadline_local']}
class DeadlineReached(Exception):pass
def guard():
 global last_guard
 if time.monotonic()-last_guard<30:return
 last_guard=time.monotonic();tool='/home/admin001/ai-video-studio/tools/gpu_lease.py';owner='lingbot-tactile-lab/real-v2-bf16-stream';s=json.loads(subprocess.check_output([sys.executable,tool,'status','--json']));assert s['state']=='held' and s['holder']==owner and s['pid_alive'];assert not any(x.get('owner')!=owner for x in s.get('waiting',[]))
 pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert all(int(p)==os.getpid() for p in pids.split())
 assert shutil.disk_usage(D).free>2*1024**3 and time.monotonic()-start<cfg_run['wall_limit_s']
 subprocess.run([sys.executable,tool,'renew','--owner',owner,'--ttl','1200'],check=True,stdout=subprocess.DEVNULL)
def checkpoint(name):
 save(D/name,{'parameters':{n:p.detach().cpu().clone() for n,p in params.items()},'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'config':cfg_run,'dataset_sha256':sha(index)})
try:
 b.require_reproducibility_gate();b.require_gpu_window();_,_,weights=b.verify_fixed_inputs(R)
 from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
 apply_lingbot_qwen3_vl_patch();torch.manual_seed(cfg_run['seed']);cfg=b.build_old_config();cfg.use_cache=False;cfg.precompute_grid_thw=False
 model,proof=b.load_model(cfg,R/'work/contact_temporal_model_v9/modeling_lingbot_vla_v2.py',weights)
 model.model.tactile_adapter=CWMJTemporalConditioner(cfg.proj_width).float().cuda();model.model.action_step_conditioner=ActionStepConditioner(cfg.proj_width).float().cuda();model.model.action_out_proj=ActionProjectionLoRA(model.model.action_out_proj,rank=8,alpha=8.);model.model.action_film=ContactActionFiLM(cfg.proj_width).float().cuda()
 for n,p in model.named_parameters():p.requires_grad_(n.startswith(PREFIXES))
 params={n:p for n,p in model.named_parameters() if p.requires_grad};assert params
 optimizer=torch.optim.AdamW(params.values(),lr=cfg_run['lr'],weight_decay=0);assert not optimizer.state
 source=Path(cfg_run['source_checkpoint']); restored=torch.load(source,map_location='cpu',weights_only=False)
 assert restored['epoch']==10 and restored['step']==20380 and restored['dataset_sha256']==sha(index)
 assert set(restored['parameters'])==set(params)
 with torch.no_grad():
  for name,p in params.items():p.copy_(restored['parameters'][name])
 optimizer.load_state_dict(restored['optimizer']);step=restored['step'];epoch=restored['epoch']
 del restored
 main=FullTrajectoryDataset(index,build_feature_transform());assert len(main.index['trajectories'])==50
 per_epoch=math.ceil(len(main)/cfg_run['batch_size']);total=10*per_epoch
 dump(O/'initialization.json',{'fresh_optimizer':False,'no_tactile_checkpoint_restored':False,'source_checkpoint':str(source),'source_sha256':sha(source),'base_manifest_sha256':sha(R/cfg_run['base_shards_manifest']),'dataset_sha256':sha(index),'trainable_names':list(params),'windows':len(main),'steps_per_epoch':per_epoch,'total_steps':total})
 checkpoint('initial.pt');model.train();torch.cuda.reset_peak_memory_stats();status['status']='training';dump(O/'status.json',status)
 for epoch_index in range(10,100000):
  order=list(range(len(main)));random.Random(cfg_run['seed']+epoch_index).shuffle(order);assert len(set(order))==len(main)
  dump(O/f'epoch_{epoch_index+1:02d}_order.json',{'indices':order,'dataset_sha256':sha(index)})
  seen=0
  for off in range(0,len(order),cfg_run['batch_size']):
   if time.time()>=cfg_run['deadline_unix']:raise DeadlineReached()
   guard();ids=order[off:off+cfg_run['batch_size']];n=len(ids);optimizer.zero_grad(set_to_none=True);loaded=[main[i] for i in ids];xs=[x['batch'] for x in loaded]
   batch={k:b.gpu(torch.stack([x[k] for x in xs]) if k=='image_grid_thw' else torch.cat([x[k] for x in xs],0)) for k in xs[0]};payloads=[x['payload'] for x in loaded];pp={k:b.gpu(torch.cat([p[k] for p in payloads],0)) for k in payloads[0]}
   torch.manual_seed(cfg_run['seed']+10000+step);noise=torch.randn_like(batch['actions']);t=model.model.sample_time(n,'cuda').to(batch['actions'].dtype);rate=cfg_run['min_lr']+.5*(cfg_run['lr']-cfg_run['min_lr'])*(1+math.cos(math.pi*step/max(1,total-1)))
   for group in optimizer.param_groups:group['lr']=rate
   output=model(**batch,**pp,noise=noise,time=t,action_step_s=.05,return_action_loss_vector=True);loss=output[6]['action_loss_per_sample_grad'].mean();assert torch.isfinite(loss);loss.backward()
   if step==0:
    gradients={n:{'present':p.grad is not None,'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),'norm':float(p.grad.float().norm()) if p.grad is not None else 0.} for n,p in params.items()};assert all(v['finite'] for v in gradients.values());assert any(v['norm']>0 for n,v in gradients.items() if n.startswith('model.tactile_adapter.tactile.'));assert any(v['norm']>0 for n,v in gradients.items() if n.startswith('model.action_film.'));dump(O/'fusion_gradient_check.json',{'passed':True,'gradients':gradients,'scope':'finite action-loss gradient through tactile and fusion; not capability proof'})
   norm=torch.nn.utils.clip_grad_norm_(list(params.values()),cfg_run['gradient_clip'],error_if_nonfinite=True);optimizer.step();step+=1;seen+=n
   stages={name:sum(x['window']['stage']==name for x in loaded) for name in {x['window']['stage'] for x in loaded}}
   first_targets=[main.actions[x['window']['seed']][x['window']['offset']] for x in loaded]
   row={'step':step,'epoch':epoch_index+seen/len(main),'completed_epochs':epoch_index,'batch_size':n,'lr':rate,'action_loss':float(loss.detach()),'gradient_norm':float(norm),'elapsed_s':time.monotonic()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'phase_counts':stages,'first_target_left_gripper_normalized':sum(float(a[6]) for a in first_targets)/n,'first_target_right_gripper_normalized':sum(float(a[13]) for a in first_targets)/n}
   with (O/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   del output,loss,batch,pp
   if step%16==0:status.update(**row);dump(O/'status.json',status)
  assert seen==len(main);epoch=epoch_index+1;checkpoint('latest.pt');status.update(completed_epochs=epoch,step=step);dump(O/'status.json',status)
 assert step==total;status.update(status='completed',completed_epochs=10,step=step)
except DeadlineReached:
 checkpoint('deadline_final.pt');status.update(status='completed',stop_reason='authorized_deadline',step=step,completed_epochs=epoch)
except BaseException as exc:
 status.update(status='failed',error=repr(exc),step=step,completed_epochs=epoch)
 if optimizer is not None:
  try:checkpoint('interrupted.pt')
  except Exception as e:status['checkpoint_error']=repr(e)
 raise
finally:
 status['elapsed_s']=time.monotonic()-start;dump(O/'status.json',status)

from pathlib import Path
import json,time,hashlib,random,subprocess,os,sys,shutil
import torch,math
from cwmj_full_trajectory_data import FullTrajectoryDataset
from cwmj_model_bridge_v1 import build_feature_transform
import run_cwmj_m7_training as b
from cwmj_temporal_conditioner import CWMJTemporalConditioner
from action_step_conditioner import ActionStepConditioner
from contact_action_projection_lora import ActionProjectionLoRA
from action_projection_capacity import ActionProjectionCapacity
from contact_action_film import ContactActionFiLM
from action_timed_contact_fraction import TimedContactFractionHead,contact_fraction_loss
from action_timed_motion import TimedActionMotionHead
from action_motion_supervision import motion_loss
from cwmj_native_policy_inference import NativeCWMJInference
from contact_policy_inference import VISUAL_KEYS
R=b.REPO;D=R/'data/cwmj_full_trajectory_extension_v1';O=R/'data/reports/cwmj_full_trajectory_extension_v1';O.mkdir(exist_ok=False);config=json.loads((R/'datasets/cwmj_full_trajectory_extension_v1.json').read_text());start=time.monotonic();status={'status':'loading','completed_arms':[]};arm=None;completed_step=0;optimizer=None;last_guard=0
PREFIXES=('model.tactile_adapter.','model.action_step_conditioner.','model.action_film.','model.action_out_proj.lora_');HEADS=('model.action_contact_head.','model.action_motion_head.')
def atomic_json(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def save(path,obj):
 tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def log(row):
 with (O/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def guard():
 global last_guard
 if time.monotonic()-last_guard<30:return
 last_guard=time.monotonic();tool='/home/admin001/ai-video-studio/tools/gpu_lease.py';owner='lingbot-tactile-lab/real-v2-bf16-stream';info=json.loads(subprocess.run([sys.executable,tool,'status','--json'],check=True,capture_output=True,text=True).stdout)
 if info.get('state')!='held' or info.get('holder')!=owner or not info.get('pid_alive') or any(w.get('owner')!=owner for w in info.get('waiting',[])):raise InterruptedError('own lease unavailable or external waiter')
 pids=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout
 if any(int(x.strip())!=os.getpid() for x in pids.splitlines() if x.strip()):raise InterruptedError('other GPU process')
 if time.time()>=config['deadline_unix']:raise InterruptedError('user deadline10:00 reached; retain partial checkpoint')
 if shutil.disk_usage(D).free<2*1024**3 or time.monotonic()-start>config['wall_limit_s']:raise InterruptedError('resource/time limit')
 subprocess.run([sys.executable,tool,'renew','--owner',owner,'--ttl','1200'],check=True,stdout=subprocess.DEVNULL)
def snapshot():return {n:p.detach().cpu().clone() for n,p in params.items()}
def checkpoint(step):
 save(D/(arm+'_latest.pt'),{'parameters':snapshot(),'optimizer':optimizer.state_dict(),'step':step,'arm':arm,'config':config});status.update(arm=arm,step=step,elapsed_s=time.monotonic()-start);atomic_json(O/'status.json',status)
def control(payload,donor,arm):
 pp={k:v.clone() for k,v in payload.items()}
 if arm=='joint_vision_proprio':pp['tactile_mask'].zero_();pp['tactile_valid'].zero_()
 if arm=='joint_shuffled_tactile':
  for k in ('tactile_history','tactile_mask','tactile_age_ms','tactile_valid'):pp[k]=donor[k].clone()
 return pp


try:
 b.require_reproducibility_gate();b.require_gpu_window();_,_,weights=b.verify_fixed_inputs(R)
 from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
 apply_lingbot_qwen3_vl_patch();torch.manual_seed(config['seed']);cfg=b.build_old_config();cfg.use_cache=False;cfg.precompute_grid_thw=False
 model,proof=b.load_model(cfg,R/'work/contact_temporal_model_v9/modeling_lingbot_vla_v2.py',weights)
 model.model.tactile_adapter=CWMJTemporalConditioner(cfg.proj_width).float().cuda();model.model.action_step_conditioner=ActionStepConditioner(cfg.proj_width).float().cuda();model.model.action_out_proj=ActionProjectionLoRA(model.model.action_out_proj,rank=8,alpha=8.);model.model.action_film=ContactActionFiLM(cfg.proj_width).float().cuda();model.model.action_contact_head=TimedContactFractionHead(cfg.proj_width).float().cuda();model.model.action_motion_head=TimedActionMotionHead(cfg.proj_width).float().cuda();torch.nn.init.zeros_(model.model.action_motion_head.head.weight);torch.nn.init.zeros_(model.model.action_motion_head.head.bias)
 for n,p in model.named_parameters():p.requires_grad_(n.startswith(PREFIXES+HEADS))
 params={n:p for n,p in model.named_parameters() if p.requires_grad};path=Path(config['source_checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==config['source_sha256'];source=torch.load(path,weights_only=True,map_location='cpu');assert source['step']==3456 and source['arm']=='joint_tactile';original=source['parameters']
 assert set(original)=={n for n in params if not n.startswith('model.action_out_proj.capacity_')}
 with torch.no_grad():
  for n,p in params.items():
   if n in original:p.copy_(original[n])
 initial=snapshot();save(D/'initial_all.pt',initial)
 index_path=R/'data/cwmj_full_trajectory_windows_v1/index.json';assert hashlib.sha256(index_path.read_bytes()).hexdigest()==config['window_index_sha256'];main=FullTrajectoryDataset(index_path,build_feature_transform());assert len(main)==18304
 manifest=json.loads((D/'manifest.json').read_text());assert hashlib.sha256((D/'aux.pt').read_bytes()).hexdigest()==manifest['aux_sha256'];aux=torch.load(D/'aux.pt',weights_only=True,map_location='cpu');items=aux['items'];assert all(not a['behavior_cloning_allowed'] for a in items)
 atomic_json(O/'sources.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'datasets/cwmj_full_trajectory_extension_v1.json',D/'manifest.json',index_path,R/'src/cwmj_full_trajectory_data.py',R/'src/action_timed_contact_fraction.py',R/'src/cwmj_native_policy_inference.py']});status['status']='training';torch.cuda.reset_peak_memory_stats()
 for arm in config['arms']:
  with torch.no_grad():
   for n,p in params.items():p.copy_(initial[n])
  for n,p in model.named_parameters():
   if p.requires_grad:assert n in params
  optimizer=torch.optim.AdamW([p for p in params.values() if p.requires_grad],lr=config['lr'],weight_decay=0.);completed_step=0;checkpoint(0)
  prior=torch.load(R/'data/cwmj_full_trajectory_training_v1'/(arm+'_latest.pt'),map_location='cpu',weights_only=True);assert prior['step']==13728 and prior['arm']==arm and set(prior['parameters'])==set(params)
  with torch.no_grad():
   for n,p in params.items():p.copy_(prior['parameters'][n])
  optimizer.load_state_dict(prior['optimizer']);checkpoint(0)
  order=[]
  for epoch in range(3,5):
   epoch_ids=list(range(len(main)));random.Random(config['seed']+epoch).shuffle(epoch_ids);order.extend(epoch_ids)
  for step in range(1,config['steps_per_arm']+1):
   guard();optimizer.zero_grad(set_to_none=True);rng=random.Random(config['seed']+step);ids=order[(step-1)*4:step*4];assert len(ids)==4;torch.manual_seed(config['seed']+20000+step);loaded=[main[i] for i in ids];xs=[x['batch'] for x in loaded];batch={k:b.gpu(torch.stack([x[k] for x in xs]) if k=='image_grid_thw' else torch.cat([x[k] for x in xs],0)) for k in xs[0]};payloads=[]
   for pos,i in enumerate(ids):
    donor=rng.randrange(len(main)-1);donor+=donor>=i;payloads.append(control(loaded[pos]['payload'],main.payload_only(donor) if arm=='joint_shuffled_tactile' else loaded[pos]['payload'],arm))
   pp={k:b.gpu(torch.cat([p[k] for p in payloads],0)) for k in payloads[0]};torch.manual_seed(config['seed']+10000+step);noise=torch.randn_like(batch['actions']);t=model.model.sample_time(4,'cuda').to(batch['actions'].dtype)
   rate=config['min_lr']+.5*(config['lr']-config['min_lr'])*(1+math.cos(math.pi*(step-1)/(config['steps_per_arm']-1)))
   for group in optimizer.param_groups:group['lr']=rate
   output=model(**batch,**pp,noise=noise,time=t,action_step_s=.05,return_action_loss_vector=True);al=output[6]['action_loss_per_sample_grad'].mean()
   pair=rng.choice(aux['pairs']);aids=pair+[rng.choice(aux['zeros']),rng.choice(aux['positives'])];aps=[];donor=rng.randrange(len(items))
   while donor in aids:donor=rng.randrange(len(items))
   for i in aids:aps.append(control(items[i]['payload'],items[donor]['payload'],arm))
   ap={k:torch.cat([p[k] for p in aps],0).cuda() for k in aps[0]};aa=torch.stack([items[i]['actions'] for i in aids]).cuda();am=torch.stack([items[i]['mask'] for i in aids]).cuda();dt=torch.stack([items[i]['durations'] for i in aids]).cuda();ct=torch.stack([items[i]['contact_target'] for i in aids]).cuda();mt=torch.stack([items[i]['motion_target'] for i in aids]).cuda();condition=model.model.tactile_adapter(*[ap[k] for k in ['tactile_history','tactile_mask','tactile_age_ms','tactile_valid','contact_proprio','command_history','command_history_mask']]);c=model.model.action_contact_head(condition,aa,am,dt,horizon_s=.1);m=model.model.action_motion_head(condition,aa,am,dt,horizon_s=.1);cl=contact_fraction_loss(c['contact_fraction_logit'],ct,c['prediction_valid']);ml=motion_loss(m['motion_prediction'],mt,m['prediction_valid'],scales=config['motion_scales']);loss=al if arm=='direct_tactile' else al+config['contact_weight']*cl+config['motion_weight']*ml
   assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(params.values()),config['gradient_clip'],error_if_nonfinite=True);optimizer.step();assert all(torch.isfinite(p).all() for p in params.values());completed_step=step
   log({'kind':'train','arm':arm,'step':step,'lr':rate,'action_loss':float(al.detach()),'contact_loss':float(cl.detach()),'motion_loss':float(ml.detach()),'loss':float(loss.detach()),'gradient_norm':float(norm),'elapsed_s':time.monotonic()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30})
   # Drop graph-bearing locals before a diagnostic or the next large forward.
   del output,condition,c,m,loss,al,cl,ml,batch,pp,ap,aa,am,dt,ct,mt
   if step%16==0:status.update(arm=arm,step=step,elapsed_s=time.monotonic()-start);atomic_json(O/'status.json',status)
   if step%256==0:checkpoint(step)
   if step in config['checkpoint_steps']:checkpoint(step);save(D/(arm+f'_step{step:05d}.pt'),snapshot())
  save(D/(arm+'_final.pt'),{n:p.detach().cpu().clone() for n,p in params.items() if n.startswith(PREFIXES)});save(D/(arm+'_aux_heads.pt'),{n:p.detach().cpu().clone() for n,p in params.items() if n.startswith(HEADS)});status['completed_arms'].append(arm);atomic_json(O/'status.json',status)
 status.update(status='completed',elapsed_s=time.monotonic()-start,matched_updates=True)
except BaseException as exc:
 status.update(status='failed',error=repr(exc),elapsed_s=time.monotonic()-start)
 if arm and optimizer is not None:
  try:checkpoint(completed_step)
  except Exception as save_exc:status['checkpoint_error']=repr(save_exc)
 raise
finally:
 atomic_json(O/'status.json',status);print(json.dumps(status),flush=True)

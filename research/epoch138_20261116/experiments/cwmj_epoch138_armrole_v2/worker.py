import ctypes,signal,os
_parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGTERM)
if os.getppid()!=_parent:raise SystemExit('parent exited')
from pathlib import Path
import json,time,torch
import run_cwmj_m7_training as b
from cwmj_temporal_conditioner import CWMJTemporalConditioner
from action_step_conditioner import ActionStepConditioner
from contact_action_projection_lora import ActionProjectionLoRA
from contact_action_film import ContactActionFiLM
from cwmj_action_preference_loss import paired_flow_preference
R=b.REPO;O=R/'data/cwmj_epoch138_armrole_v2/worker_report';O.mkdir(exist_ok=False);result={'status':'running','optimizer_steps':0};start=time.monotonic()
try:
 b.require_reproducibility_gate();b.require_gpu_window();_,_,weights=b.verify_fixed_inputs(R)
 from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
 apply_lingbot_qwen3_vl_patch();torch.manual_seed(17011);cfg=b.build_old_config();cfg.use_cache=False;cfg.precompute_grid_thw=False
 model,proof=b.load_model(cfg,R/'work/contact_temporal_model_v9/modeling_lingbot_vla_v2.py',weights)
 model.model.tactile_adapter=CWMJTemporalConditioner(cfg.proj_width).float().cuda();model.model.action_step_conditioner=ActionStepConditioner(cfg.proj_width).float().cuda();model.model.action_out_proj=ActionProjectionLoRA(model.model.action_out_proj,rank=8,alpha=8.);model.model.action_film=ContactActionFiLM(cfg.proj_width).float().cuda()
 prefixes=['model.tactile_adapter.','model.action_step_conditioner.','model.action_film.','model.action_out_proj.lora_']
 for n,p in model.named_parameters():p.requires_grad_(any(n.startswith(x) for x in prefixes))

 import sys,numpy as np,hashlib
 from cwmj_native_policy_inference import NativeCWMJInference
 from cwmj_full_trajectory_data import causal_payload as payload
 from cwmj_model_bridge_v1 import build_feature_transform,decode_model_actions
 from contact_policy_inference import prepare_observation
 from prepare_real_vla_visual_smoke import CAMERAS
 session=Path(sys.argv[1]);transform=build_feature_transform();contract=json.loads((R/'datasets/cwmj_aloha_14d_contract_v3_1.json').read_text());engine=NativeCWMJInference(model)
 audit=json.loads((R/'work/cwmj_epoch138_armrole_v2/checkpoint.json').read_text());params={n:p for n,p in model.named_parameters() if p.requires_grad};current=None
 (session/'ready.json').write_text('{}');index=0
 while not (session/'stop.json').exists():
  request=session/f'request_{index:04d}.json'
  if not request.exists():time.sleep(.05);continue
  req=json.loads(request.read_text());arm=req['arm']
  if arm!=current:
   entry=audit;path=Path(entry['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['step']==281244 and saved['epoch']==138;state=saved['parameters'];assert params.keys()==state.keys()
   with torch.no_grad():
    for n,p in params.items():p.copy_(state[n])
   current=arm
  path=Path(req['metadata']);assert hashlib.sha256(path.read_bytes()).hexdigest()==req['metadata_sha256'];meta=json.loads(path.read_text())
  with np.load(path.parent/meta['images_file']) as images:rgb={k:images[v.split('.')[-1]] for k,v in CAMERAS.items()}
  visual,metadata=prepare_observation(transform,images_rgb=rgb,command_state_14=meta['observation']['command_history'][-1]['action_14d_normalized'],instruction=meta['observation']['language']['instruction'])
  pp=payload(meta)
  if req['mode']=='masked':pp['tactile_mask'].zero_();pp['tactile_valid'].zero_()
  elif req['mode']=='donor':
   donor_path=Path(req['donor']['path']);assert hashlib.sha256(donor_path.read_bytes()).hexdigest()==req['donor']['sha256'];donor=torch.load(donor_path,map_location='cpu',weights_only=True)['payload']
   for k in ('tactile_history','tactile_mask','tactile_age_ms','tactile_valid'):pp[k]=donor[k].clone()
  sample=engine.predict(visual,pp,noise_seed=req['noise_seed']);decoded=decode_model_actions(transform,metadata,sample['actions'][0],contract=contract)
  if req.get('audit_language'):
   original_instruction=meta['observation']['language']['instruction'];audit_rows=[]
   baseline=np.asarray(decoded.get('commands_14d',decoded.get('raw_commands_14d')))
   for alternative in ['Use the left arm to lift, hold, and place back the tomato sauce can; keep the other arm clear.','Use the right arm to lift, hold, and place back the tomato sauce can; keep the other arm clear.']:
    if alternative==original_instruction:continue
    other_visual,other_metadata=prepare_observation(transform,images_rgb=rgb,command_state_14=meta['observation']['command_history'][-1]['action_14d_normalized'],instruction=alternative)
    other=engine.predict(other_visual,pp,noise_seed=req['noise_seed']);other_decoded=decode_model_actions(transform,other_metadata,other['actions'][0],contract=contract)
    values=np.asarray(other_decoded.get('commands_14d',other_decoded.get('raw_commands_14d')))
    audit_rows.append({'instruction':alternative,'language_tokens_differ':not torch.equal(visual['lang_tokens'],other_visual['lang_tokens']),'decoded':other_decoded,'first_action_difference':(values[0]-baseline[0]).tolist(),'same_observation':str(path),'same_observation_sha256':req['metadata_sha256'],'same_noise_seed':req['noise_seed'],'scope':'offline counterfactual prediction only; no execution'})
   (session/f'language_audit_{index:04d}.json').write_text(json.dumps({'original_instruction':original_instruction,'alternatives':audit_rows},indent=2))
  answer={'decoded':decoded,'elapsed_s':sample['elapsed_s'],'observation_time_s':meta['observation']['decision_time_s'],'arm':arm,'noise_seed':req['noise_seed']}
  output=session/f'response_{index:04d}.json';tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(answer));tmp.replace(output);index+=1
 result.update(status='completed',requests=index,optimizer_steps=0)
except Exception as exc:result.update(status='failed',error=repr(exc));raise
finally:
 result['wall_s']=time.monotonic()-start;(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

from pathlib import Path
import json,time,subprocess,hashlib,os,sys,shutil
from weighted_selection import select_weighted
from arm_role import apply_role
from alignment_guard import AlignmentGuard
import numpy as np,mujoco
from PIL import Image,ImageDraw
from cwmj_fixed_dual_arm_platform import FixedDualArmScenarioEnv
from cwmj_mainline_env import MainlineContinuousEnv
from cwmj_monitored_execution import MonitoredContinuousTargetKernel
from cwmj_multimodal_capture import capture_observation
from mainline_gripper_projection import admit_projected_first_target
from metrics import evaluate_full_task
R=Path('/mnt/relaydata/lingbot-tactile-lab');W=R/'work/cwmj_epoch138_alignguard_v2';D=R/'data/cwmj_epoch138_alignguard_v2';S=D/'session';S.mkdir();plan=json.loads((W/'plan.json').read_text());summary={'status':'running','planned':4,'episodes':[],'training_steps':0};index=0;worker=None;env=None
lease='/home/admin001/ai-video-studio/tools/gpu_lease.py';owner='lingbot-tactile-lab/real-v2-bf16-stream'
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 info=json.loads(subprocess.check_output([sys.executable,lease,'status','--json']));assert info['state']=='held' and info['holder']==owner and info['pid_alive'];assert not any(x.get('owner')!=owner for x in info.get('waiting',[]))
 pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert all(int(x) in [worker.pid,os.getpid()] for x in pids.split());assert shutil.disk_usage(D).free>5*1024**3
 subprocess.run([sys.executable,lease,'renew','--owner',owner,'--ttl','1200'],check=True,stdout=subprocess.DEVNULL)
def video(d,case,records):
 frames=[d/'reset_images.npz']+[d/f'observation_{i:03d}_images.npz' for i in range(sum(bool(x.get('diagnostic_file')) for x in records))]
 p=subprocess.Popen(['/home/admin001/miniconda3/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x272','-r','20','-i','-','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(d/'rollout.mp4')],stdin=subprocess.PIPE)
 for i,f in enumerate(frames):
  im=Image.new('RGB',(960,272),'white');ImageDraw.Draw(im).text((5,5),f'MODEL EVAL epoch138 | {case["side"]} seed{case["seed"]} repeat{case["repeat"]} | frame{i} 1x',fill='black')
  with np.load(f) as z:
   for j,k in enumerate(['cam_high','cam_left_wrist','cam_right_wrist']):im.paste(Image.fromarray(z[k]),(320*j,32))
  if i in [0,len(frames)-1]:im.save(d/f'frame_{i:03d}.png')
  p.stdin.write(np.asarray(im).tobytes())
 p.stdin.close();assert p.wait()==0
 subprocess.run(['/home/admin001/miniconda3/bin/ffmpeg','-v','error','-i',str(d/'rollout.mp4'),'-f','null','-'],check=True)
 write(d/'completion.json',{'status':'capture_complete','frames':len(frames),'video_sha256':sha(d/'rollout.mp4'),'training_admitted':False,'model_evaluation':True})
 subprocess.run([sys.executable,str(R/'scripts/build_push_force_review.py'),'cwmj_epoch138_alignguard_v2',d.name],check=True)
def publish(d):
 portal=Path('/home/admin001/lingbot-tactile-artifacts/vtla-review');link=portal/'cwmj_epoch138_alignguard_v2'
 if not link.exists():link.symlink_to(D.resolve())
 html='<meta charset="utf-8"><h2>Epoch 138 · 加权时间集成 · 横向到位约束 · 4回合对照</h2><p>固定双臂相机；完整视频与同步力表。进行中，失败如实保留。</p>'
 for row in summary['episodes']:html+=f'<p><a href="{row["case"]}/force_review.html">{row["case"]}</a> — {row["termination"]}; success={row["metrics"]["qualified_task_success"]}</p>'
 (D/'index.html').write_text(html)
 top=portal/'index.html';s=top.read_text()
 if 'cwmj_epoch138_alignguard_v2/index.html' not in s:top.write_text('<p><a href="/cwmj_epoch138_alignguard_v2/index.html">Epoch 138 模型闭环评测 · 视频与同步力表</a></p>'+s)
try:
 worker=subprocess.Popen(['/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python','-u',str(W/'worker.py'),str(S)],cwd=R,stdout=(S/'worker.log').open('w'),stderr=subprocess.STDOUT)
 deadline=time.monotonic()+360
 while not (S/'ready.json').exists():
  assert worker.poll() is None,'worker load failed';assert time.monotonic()<deadline;time.sleep(.2)
 for position,case in enumerate(plan['episodes_order']):
  guard();key=case['case_id']+'_r'+str(case['repeat'])+'__'+case['ensemble']['name'];out=D/key;out.mkdir();records=[];reason='horizon_complete';technical_error=None;prediction_history=[];aligner=None
  scenario={'namespace':'fixed_dual_reserved_test','seed':case['seed'],'xy_offset_m':case['offset_m'],'yaw_offset_rad':0}
  env=MainlineContinuousEnv(FixedDualArmScenarioEnv(scenario,active_side=case['side'],render=True,platform_path=R/plan['platform']));env.kernel.__class__=MonitoredContinuousTargetKernel;env.kernel.allowed_contacts=frozenset({'asset_table','finger_object'});aligner=AlignmentGuard(env.env.model,case['side'])
  assert all(renderer._model is env.env.model for renderer in env.env._renderers.values())
  write(out/'identity.json',env.env.trace_identity());write(out/'protocol.json',{'case':case,'checkpoint':json.loads((W/'checkpoint.json').read_text()),'plan_sha256':sha(W/'plan.json'),'renderer_binding_verified':True,'training_admitted':False});capture_observation(env,out,'reset');origin=env.env._force_sample_substep()['diagnostic_trace']
  try:
   ready=[]
   for arm in ['fl','fr']:ready += [env.env.platform_config['ready_arm_joints'][f'{arm}_joint{i}'] for i in range(1,7)]+[.045]
   obs,t=env.checked_step(ready);assert t['accepted'],'ready warmup rejected';diag='diagnostic_warmup.json';write(out/diag,env.env._last_interval_substeps);records.append({'stage':'ready_warmup','execution':t,'diagnostic_file':diag});capture_observation(env,out,'observation_000',observation=obs)
   origin=env.env._force_sample_substep()['diagnostic_trace'];write(out/'origin.json',origin);pre=out/'observation_000.json'
   for tick in range(plan['horizon_model_ticks']):
    if tick%10==0:guard()
    chunk_offset=0
    if chunk_offset==0:
     physical_time=float(env.env.data.time);req={'arm':'epoch138','mode':'intact','donor':None,'metadata':str(pre),'metadata_sha256':sha(pre),'noise_seed':900000+case['seed']*200+case['repeat']*10000000+tick,'audit_language':False};write(S/f'request_{index:04d}.json',req);answer_path=S/f'response_{index:04d}.json';deadline=time.monotonic()+180
     while not answer_path.exists():
      assert worker.poll() is None,'worker exited';assert time.monotonic()<deadline,'inference timed out';time.sleep(.02)
     answer=json.loads(answer_path.read_text());assert abs(answer['observation_time_s']-physical_time)<1e-9 and env.env.data.time==physical_time
     inference_pre=pre;source_tick=tick;index+=1
    prediction_history.append({'tick':tick,'decoded':answer['decoded'],'response_file':str(answer_path),'response_sha256':sha(answer_path),'metadata_file':str(pre),'metadata_sha256':sha(pre)})
    prediction_history=prediction_history[-case['ensemble']['window']:]
    admission=select_weighted(prediction_history,tick,case['ensemble'],env.env.validate_action)
    row={'stage':'model_closedloop','index':tick,'response_file':str(answer_path),'admission':admission,'inference_s':answer['elapsed_s'] if chunk_offset==0 else 0.,'pre_metadata':str(pre),'pre_metadata_sha256':sha(pre),'inference_metadata':str(inference_pre),'inference_metadata_sha256':sha(inference_pre),'chunk_source_tick':source_tick,'chunk_offset':chunk_offset,'observation_age_s':float(env.env.data.time)-answer['observation_time_s']};records.append(row)
    if not admission['accepted']:row['execution']={'accepted':False};reason='target_rejected';break
    instruction=json.loads(pre.read_text())['observation']['language']['instruction']
    role_command,role_audit=apply_role(admission['command'],instruction,ready,env.env.validate_action);row['arm_role']=role_audit
    measured=[]
    for prefix in ['fl','fr']:
     for jn in range(1,8):
      jid=mujoco.mj_name2id(env.env.model,mujoco.mjtObj.mjOBJ_JOINT,f'{prefix}_joint{jn}');measured.append(float(env.env.data.qpos[env.env.model.jnt_qposadr[jid]]))
    role_command,guard_audit=aligner.apply(role_command,measured,env.env.last_force_interval,tick);row['alignment_guard']=guard_audit
    if role_command is None:row['execution']={'accepted':False,'actual_advanced':False};reason='alignment_guard_rejected';break
    obs,t=env.checked_step(role_command,gripper_units='normalized');row['execution']=t
    if not t['accepted']:reason='physics_preflight_rejected';break
    name=f'diagnostic_{tick:03d}.json';write(out/name,env.env._last_interval_substeps);row['diagnostic_file']=name;capture_observation(env,out,f'observation_{tick+1:03d}',observation=obs);pre=out/f'observation_{tick+1:03d}.json'
    write(out/'progress.json',{'accepted_model_ticks':tick+1,'sim_time':float(env.env.data.time)})
  except Exception as exc:technical_error=repr(exc);reason='technical_error'
  write(out/'result.json',{'records':records,'termination_reason':reason,'technical_error':technical_error,'split':'development_reused_test_cases','model_evaluation':True})
  support=mujoco.mj_name2id(env.env.model,mujoco.mjtObj.mjOBJ_BODY,env.env.table_body_name);sg=np.flatnonzero(env.env.model.geom_bodyid==support).tolist()
  metrics=evaluate_full_task(out,origin,sg,plan['task_thresholds'],active_side=case['side']);write(out/'metrics.json',metrics);env.close();env=None;video(out,case,records)
  summary['episodes'].append({'case':key,'side':case['side'],'termination':reason,'metrics':metrics});write(D/'summary.json',summary);publish(out)
  if technical_error:raise RuntimeError(technical_error)
  if position==1:write(D/'small_panel_technical_gate.json',{'passed':True,'cases':[x['case'] for x in summary['episodes']],'scope':'valid inference/capture/render binding/video decode/synchronous force table; task success not required; no policy tuning'})
 summary['status']='completed';summary['successes']=sum(x['metrics']['qualified_task_success'] for x in summary['episodes'])
except BaseException as exc:summary.update(status='failed',error=repr(exc));raise
finally:
 write(D/'summary.json',summary)
 if env:env.close()
 if worker:
  write(S/'stop.json',{})
  try:worker.wait(timeout=20)
  except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=20)

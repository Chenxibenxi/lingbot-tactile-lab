"""Opt-in fixed dual-arm candidate; never replaces legacy runs in place."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco
from cwmj_mainline_scenario import ScenarioTomatoEnv
R=Path(__file__).resolve().parents[1]
CONFIG=R/'datasets/cwmj_fixed_dual_arm_platform_v1.json'
class FixedDualArmScenarioEnv(ScenarioTomatoEnv):
 def __init__(self,scenario,*,active_side='left',render=False,platform_path=CONFIG):
  if active_side not in ('left','right'):raise ValueError('explicit active side required')
  super().__init__(scenario,cameras46=True,render=False)
  self.platform_path=Path(platform_path);cfg=json.loads(self.platform_path.read_text());root=ET.fromstring(self.xml_text)
  for c in cfg['cameras']:
   nodes=root.findall(".//camera[@name='"+c['policy_key']+"']")
   if len(nodes)!=1:raise ValueError('camera binding missing/ambiguous')
   node=nodes[0];parent=next(p for p in root.iter() if node in list(p));actual='world' if parent.tag=='worldbody' else parent.get('name')
   if actual!=c['parent_body']:raise ValueError('camera mount mismatch')
   node.set('pos',' '.join(map(str,c['mjcf_pos'])));node.set('quat',' '.join(map(str,c['mjcf_quat_wxyz'])));node.set('fovy',str(c['fovy_deg']))
  self.xml_text=ET.tostring(root,encoding='unicode');self.model=mujoco.MjModel.from_xml_string(self.xml_text);self.data=mujoco.MjData(self.model);self._verify_model_readback();self.camera_contract=json.loads((R/'datasets/cwmj_camera_observation_contract_v1.json').read_text());self.platform_config=cfg;self.active_side=active_side
  old=dict(zip(self.v31['home_state']['qpos_order'],self.tomato_initial['robot_qpos_16']));old.update(cfg['ready_arm_joints'])
  for name in ['fl_joint7','fl_joint8','fr_joint7','fr_joint8']:old[name]=cfg['gripper_ready_physical_m']
  self.tomato_initial['robot_qpos_16']=[old[n] for n in self.v31['home_state']['qpos_order']];self.tomato_initial['asset_qpos_7'][0]=cfg['task_workspace']['nominal_x'];self.tomato_initial['asset_qpos_7'][1]=cfg['task_workspace'][active_side+'_y'];self.render_enabled=render
  self.instruction=f'Use the {active_side} arm to lift, hold, and place back the tomato sauce can; keep the other arm clear.'
  self.reset(seed=0)
 def trace_identity(self):
  result=super().trace_identity();result.update(platform_config_sha256=hashlib.sha256(self.platform_path.read_bytes()).hexdigest(),fixed_camera_reference_sha256=self.platform_config['camera_reference_sha256'],active_side=self.active_side,training_admitted=False);return result

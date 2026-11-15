"""Opt-in native asset replacement. Internal target alias preserved for diagnostics.
Native geometry/appearance/inertia retained; no online object-pose policy input.
"""
from pathlib import Path
import copy,json,hashlib,itertools,xml.etree.ElementTree as ET
import numpy as np,mujoco
from cwmj_fixed_dual_arm_platform import FixedDualArmScenarioEnv
ASSETS=Path('/home/admin001/robotics_ws/LIBERO/libero/libero/assets/stable_hope_objects')
class PushAssetVariantEnv(FixedDualArmScenarioEnv):
 def __init__(self,scenario,*,asset_id='tomato_sauce',**kwargs):
  if asset_id not in {'tomato_sauce','alphabet_soup','cream_cheese'}:raise ValueError('asset not reviewed for this pilot')
  requested_render=kwargs.get('render',False);init_kwargs=dict(kwargs);init_kwargs['render']=False
  super().__init__(scenario,**init_kwargs);self.actual_asset_id=asset_id;self.asset_source_hashes={};self.contact_height_delta_m=0.
  if self._renderers:raise RuntimeError('renderer created before asset model finalized')
  if asset_id=='tomato_sauce':
   self.render_enabled=requested_render
   return
  native_path=ASSETS/asset_id/(asset_id+'.xml');native=ET.parse(native_path).getroot();root=ET.fromstring(self.xml_text)
  body=native.find(".//body[@name='object']")
  if body is None:raise ValueError('native object body missing')
  dest=root.find(".//body[@name='tomato_sauce']");free=copy.deepcopy(dest.find('freejoint'))
  if free is None:raise ValueError('target freejoint missing')
  old=self._bounds(self.model,self.data)
  for child in list(dest):dest.remove(child)
  dest.append(free)
  assets=root.find('asset');mapping={}
  self.asset_source_hashes[str(native_path)]=hashlib.sha256(native_path.read_bytes()).hexdigest()
  for node in native.find('asset'):
   n=copy.deepcopy(node);oldname=n.get('name');newname='push43_'+asset_id+'_'+oldname;mapping[oldname]=newname;n.set('name',newname)
   if n.get('file'):
    p=(native_path.parent/n.get('file')).resolve();self.asset_source_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();n.set('file',str(p))
   assets.append(n)
  for node in list(assets):
   if (node.get('name') or '').startswith('push43_'):
    for key in ['texture','material','mesh']:
     if node.get(key) in mapping:node.set(key,mapping[node.get(key)])
  for node in body:
   if node.tag not in {'geom','inertial'}:raise ValueError('unsupported native child '+node.tag)
   n=copy.deepcopy(node)
   for key in ['mesh','material']:
    if n.get(key) in mapping:n.set(key,mapping[n.get(key)])
   if n.get('name'):n.set('name','push43_'+asset_id+'_'+n.get('name'))
   dest.append(n)
  self.xml_text=ET.tostring(root,encoding='unicode');self.model=mujoco.MjModel.from_xml_string(self.xml_text);self.data=mujoco.MjData(self.model);self._verify_model_readback()
  self.render_enabled=False;self.reset(seed=0);new=self._bounds(self.model,self.data)
  # Design-time support placement: native collision bottom on same support height.
  dz=old[0][2]-new[0][2];self.tomato_initial['asset_qpos_7'][2]+=float(dz);self.reset(seed=0);new=self._bounds(self.model,self.data)
  self.contact_height_delta_m=float(new[1][2]-old[1][2]);self.render_enabled=kwargs.get('render',False)
 def _bounds(self,m,d):
  mujoco.mj_forward(m,d);points=[]
  for g in self.asset_geom_ids:
   if not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
   if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('pilot only supports native box collision decomposition')
   v=np.array(list(itertools.product([-1,1],repeat=3)))*m.geom_size[g];points.extend(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
  p=np.array(points);return p.min(0),p.max(0)
 def _render_policy_images(self):
  if any(getattr(renderer,'_model',None) is not self.model for renderer in self._renderers.values()):
   raise RuntimeError('renderer/physics model mismatch: refuse policy RGB capture')
  return super()._render_policy_images()
 def trace_identity(self):
  x=super().trace_identity();x.update(actual_asset_id=self.actual_asset_id,internal_target_body_alias='tomato_sauce',asset_source_sha256=self.asset_source_hashes,contact_height_delta_m=self.contact_height_delta_m,asset_mass_kg=float(self.model.body_mass[self.asset_body_id]));return x

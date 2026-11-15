"""Development shared substep kernel. A3 is restricted to its frozen action.

Not a policy planner or an accepted environment replacement. Caller must integrate
sensor/history publication; no rendering, worker, or model side effects here.
"""
import copy
from types import SimpleNamespace
import numpy as np
import mujoco
import cwmj49_arm_trajectory as a49
from cwmj46_closure_control import ClosureGovernor, GOVERNOR

FINGER_BODIES=('fl_link7','fl_link8','fr_link7','fr_link8')

def apply_b1_contact(model):
    ids=[]
    for name in FINGER_BODIES:
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if body<0:raise ValueError('missing finger body '+name)
        geoms=[i for i in range(model.ngeom) if model.geom_bodyid[i]==body]
        if len(geoms)!=1:raise ValueError('B1 expects one geom per finger body')
        ids.extend(geoms)
    for i in ids:
        if not np.allclose(model.geom_solref[i],[.02,1.],rtol=0,atol=1e-12) or not np.allclose(model.geom_solimp[i],[.9,.95,.001,.5,2.],rtol=0,atol=1e-12):raise ValueError('unexpected baseline finger contact')
    for i in ids:
        model.geom_solref[i]=[.004,1.];model.geom_solimp[i]=[.99,.999,.001,.5,2.]
    return [{'body':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])),'geom':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i),'runtime_id':i} for i in ids]

class FrozenA3Kernel:
    def __init__(self,env,inputs,reference):
        if not np.isclose(env.model.opt.timestep,.002):raise ValueError('physics clock mismatch')
        self.env=env;self.reference=copy.deepcopy(reference);self.start_time=float(env.data.time)
        self.raw=np.asarray(inputs['physical'],dtype=float).copy()
        self.reset_ctrl=env.data.ctrl.copy();self.ids=a49.joint_ids(env.model);self.acts=a49.actuator_ids(env.model)
        self.governor=ClosureGovernor(GOVERNOR);self.force_prev=a49.finger_force(env)
        self.steps=0;self.history=[];self.stopped=None;self.allowed_contacts=frozenset({'asset_table'})
    def replace_target(self,target):
        if not np.array_equal(np.asarray(target),self.raw):raise ValueError('A3 fixed waypoint is not a general policy action executor')
    def clone(self):
        data=mujoco.MjData(self.env.model);mujoco.mj_copyData(data,self.env.model,self.env.data)
        result=copy.copy(self)
        result.env=SimpleNamespace(model=self.env.model,data=data,asset=copy.deepcopy(self.env.asset),actuator_ids=copy.deepcopy(self.env.actuator_ids),finger_body_ids=copy.deepcopy(self.env.finger_body_ids))
        for name in ['reference','raw','reset_ctrl','ids','acts','governor','force_prev','history']:setattr(result,name,copy.deepcopy(getattr(self,name)))
        return result
    def substep(self):
        if self.stopped:raise RuntimeError(self.stopped)
        env=self.env;d=env.data;m=env.model
        ref=self.reference.sample(float(d.time)-self.start_time);effective=self.raw.copy()
        arm_indices=[0,1,2,3,4,5,7,8,9,10,11,12]
        effective[arm_indices]=ref['position']
        ctrl=self.reset_ctrl.copy()
        for i,name in enumerate(a49.ARM_JOINTS):ctrl[env.actuator_ids[name]]=ref['position'][i]
        for side,prefix,idx,force in [('left','fl',6,self.force_prev[0]),('right','fr',13,self.force_prev[1])]:
            main=self.ids[prefix+'_joint7'];mimic=self.ids[prefix+'_joint8']
            decision=self.governor.decide(side,float(self.raw[idx]),float(d.qpos[main['qpos']]),float(d.qpos[mimic['qpos']]),float(d.qvel[main['dof']]),force,{'control_tick':self.steps//25,'substep':self.steps%25})
            effective[idx]=decision['effective_target_m'];ctrl[self.acts[prefix+'_gripper']]=effective[idx]
        d.ctrl[:]=ctrl;mujoco.mj_step(m,d);self.steps+=1;self.force_prev=a49.finger_force(env)
        q=np.array([d.qpos[self.ids[n]['qpos']] for n in a49.FINGER_JOINTS])
        unexpected=[c for c in a49.contact_classes(env) if c not in self.allowed_contacts]
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():self.stopped='nonfinite'
        elif sum(int(w.number) for w in d.warning):self.stopped='solver_warning'
        elif unexpected:self.stopped='unexpected_contact'
        elif (q<-.000045).any() or (q>.04765+.000045).any():self.stopped='finger_limit'
        row={'time_s':float(d.time),'raw':self.raw.tolist(),'reference_arm':ref['position'].tolist(),'effective':effective.tolist(),'mimic_residual_m':float(max(abs(q[0]-q[1]),abs(q[2]-q[3]))),'stop':self.stopped}
        self.history.append(row)
        return row

class ContinuousTargetKernel(FrozenA3Kernel):
    """Direct smooth reference, no A3 waypoint and no collision avoidance claim.

Every new target must still undergo whole-state prefix preflight by the caller.
Position/velocity continuity refers to the reference, not physical tracking.
"""
    def __init__(self,env):
        arm=[float(env.data.qpos[a49.joint_ids(env.model)[n]['qpos']]) for n in a49.ARM_JOINTS]
        reference=a49.SyncArmTrajectory(12);reference.reset(arm)
        raw=np.zeros(14);raw[[0,1,2,3,4,5,7,8,9,10,11,12]]=arm
        for i,n in [(6,'fl_joint7'),(13,'fr_joint7')]:raw[i]=float(env.data.qpos[a49.joint_ids(env.model)[n]['qpos']])
        super().__init__(env,{'physical':raw},reference)
        # reset() stores p0 but sample() reads coeffs; initialize an explicit hold.
        self.reference.set_target(arm,0.,current_p=arm,current_v=np.zeros(12))
        self.replacements=[]
    def replace_target(self,target):
        target=np.asarray(target,dtype=float)
        if target.shape!=(14,) or not np.isfinite(target).all():raise ValueError('finite 14D target required')
        if np.array_equal(target,self.raw):return
        now=float(self.env.data.time)-self.start_time
        before=self.reference.sample(now)
        candidate=copy.deepcopy(self.reference)
        candidate.set_target(target[[0,1,2,3,4,5,7,8,9,10,11,12]],now,current_p=before['position'],current_v=before['velocity'])
        after=candidate.sample(now)
        if not np.allclose(before['position'],after['position'],atol=1e-12,rtol=0) or not np.allclose(before['velocity'],after['velocity'],atol=1e-12,rtol=0):raise ValueError('target replacement discontinuity')
        self.reference=candidate;self.raw=target.copy()
        self.replacements.append({'time_s':now,'position_jump':float(np.max(np.abs(before['position']-after['position']))),'velocity_jump':float(np.max(np.abs(before['velocity']-after['velocity']))),'target':target.tolist()})
    def clone(self):
        result=super().clone();result.replacements=copy.deepcopy(self.replacements);return result

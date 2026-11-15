"""Opt-in arm-position monitoring; measured dynamics are diagnostics, not calibrated limits."""
import numpy as np
import mujoco
import cwmj49_arm_trajectory as a49
from cwmj_mainline_execution import ContinuousTargetKernel

class MonitoredContinuousTargetKernel(ContinuousTargetKernel):
    position_tolerance_rad=1e-6  # Explicit numerical tolerance, not a hardware allowance.
    def arm_state(self):
        m,d=self.env.model,self.env.data;rows=[]
        for name in a49.ARM_JOINTS:
            jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
            if jid<0:raise ValueError('missing arm joint '+name)
            q=float(d.qpos[m.jnt_qposadr[jid]]);v=float(d.qvel[m.jnt_dofadr[jid]]);acc=float(d.qacc[m.jnt_dofadr[jid]])
            lo,hi=map(float,m.jnt_range[jid]);finite=bool(np.isfinite([q,v,acc]).all())
            excursion=max(lo-q,q-hi,0.) if m.jnt_limited[jid] else 0.
            rows.append({'joint':name,'position_rad':q,'velocity_rad_s':v,'acceleration_rad_s2':acc,'finite':finite,'limited':bool(m.jnt_limited[jid]),'bounds_rad':[lo,hi],'excursion_rad':excursion})
        return rows
    def substep(self):
        before=self.arm_state()
        if any(not x['finite'] or x['excursion_rad']>self.position_tolerance_rad for x in before):
            raise ValueError('initial measured arm state outside monitored contract')
        row=super().substep();after=self.arm_state()
        if any(not x['finite'] for x in after):self.stopped=self.stopped or 'arm_nonfinite'
        elif any(x['excursion_rad']>self.position_tolerance_rad for x in after):self.stopped=self.stopped or 'arm_joint_limit'
        row.update(stop=self.stopped,arm_state=after,arm_position_tolerance_rad=self.position_tolerance_rad)
        return row

"""Passive diagnostic snapshot; never substitute these fields into policy inputs."""
import numpy as np


def snapshot(robot):
    result={}
    for side in ('left','right'):
        entity=getattr(robot,side+'_entity')
        active=list(entity.get_active_joints())
        qpos=np.asarray(entity.get_qpos(),dtype=float)
        indices=[]
        offset=0
        for joint in active:
            dof=int(joint.get_dof())
            indices.append((joint,offset,dof));offset+=dof
        if offset!=len(qpos):raise ValueError('joint DOF mapping does not match qpos')
        arm=list(getattr(robot,side+'_arm_joints'))
        physical=[];targets=[]
        for joint in arm:
            match=[(i,d) for j,i,d in indices if j is joint]
            if len(match)!=1 or match[0][1]!=1:raise ValueError('unique one-DOF arm joint required')
            physical.append(float(qpos[match[0][0]]))
            target=np.asarray(joint.get_drive_target()).reshape(-1)
            if len(target)!=1:raise ValueError('one drive target required')
            targets.append(float(target[0]))
        if not np.isfinite(qpos).all() or not np.isfinite(targets).all():raise ValueError('nonfinite physical/target state')
        result[side]=dict(arm_actual_qpos=physical,arm_drive_targets=targets,
            entity_actual_qpos=qpos.tolist(),
            active_joint_names=[j.get_name() for j in active],
            active_joint_dofs=[d for _,_,d in indices],
            gripper_control_value=float(getattr(robot,'get_'+side+'_gripper_val')()))
    return result

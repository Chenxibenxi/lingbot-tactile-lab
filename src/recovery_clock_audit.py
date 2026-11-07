"""Recovery action/physics/camera alignment, separate from training approval."""
from recovery_supervision_contract import recovery_target_mask


def audit_clock(actions,physics,frames,observations,*,official_success):
    contract=recovery_target_mask(actions,official_success=official_success)
    if not actions or len(frames)!=len(actions):raise ValueError('one camera frame per action required')
    for i,row in enumerate(physics,1):
        if type(row.get('physics_step')) is not int or row['physics_step']!=i:
            raise ValueError('raw physics must contain every step without gaps')
    if actions[0]['before_physics_step']!=0 or actions[-1]['after_physics_step']!=len(physics):
        raise ValueError('action coverage must span the full recorded physics clock')
    for i,(action,frame) in enumerate(zip(actions,frames)):
        before=action['before_physics_step'];index=frame.get('obs_index')
        if (type(action.get('camera_frame_index')) is not int or action['camera_frame_index']!=i
                or type(frame.get('frame_index')) is not int or frame['frame_index']!=i
                or type(frame.get('physics_step')) is not int or frame['physics_step']!=before
                or type(index) is not int or not 0<=index<len(observations)
                or observations[index].get('obs_index')!=index
                or type(observations[index].get('physics_step')) is not int
                or observations[index]['physics_step']!=before
                or action.get('controller_method')!='Base_Task.take_action'
                or action.get('training_approval') is not False):
            raise ValueError('same-run camera/controller/physics anchor mismatch')
    return dict(action_count=len(actions),physics_steps=len(physics),
                external_interventions=sum(x['source_role']=='external_intervention' for x in actions),
                target_contract=contract,clock_alignment_passed=True,training_approval=False,
                still_requires=['raw headers and SHA provenance','HDF5 image/action alignment',
                                'validate_expert_data','official completion audit'])

"""Opt-in first-target execution scope; never reinterpret whole-chunk acceptance."""
import numpy as np

GRIPPER_REASON='physical gripper target outside [0,1]; explicit clipping required'

def admit_first_target(decoded,validate_action):
    status=decoded.get('status')
    if status=='DECODED':raw=decoded.get('commands_14d')
    elif status=='DECODE_REJECTED' and decoded.get('reason')==GRIPPER_REASON:
        raw=decoded.get('raw_commands_14d')
    else:return {'accepted':False,'reason':'unsupported_or_failed_decode'}
    try:a=np.asarray(raw,dtype=np.float64)
    except (ValueError,TypeError):return {'accepted':False,'reason':'invalid_decoded_array'}
    if a.shape!=(50,14) or not np.isfinite(a).all():return {'accepted':False,'reason':'shape_or_nonfinite'}
    bad=np.argwhere((a[:,[6,13]]<0)|(a[:,[6,13]]>1))
    diagnostics=[{'row':int(i),'dimension':[6,13][int(j)],'value':float(a[i,[6,13][j]])} for i,j in bad]
    if any(x['row']==0 for x in diagnostics):return {'accepted':False,'reason':'first_target_gripper_invalid','chunk_violations':diagnostics}
    try:validated=validate_action(a[0].tolist(),gripper_units='normalized')
    except ValueError as exc:return {'accepted':False,'reason':'first_target_validation','detail':str(exc),'chunk_violations':diagnostics}
    return {'accepted':True,'contract':'first_target_replan_v1','command':a[0].tolist(),'validated':validated,'chunk_violations':diagnostics,'whole_chunk_executable':not diagnostics and status=='DECODED','clipping_applied':False,'executed_source_row':0,'requires_physics_preflight':True}

"""Explicit actuator-domain projection for one-step replanning, with raw audit."""
import copy
import numpy as np
from mainline_first_target_contract import admit_first_target, GRIPPER_REASON

def admit_projected_first_target(decoded, validate_action):
    status=decoded.get('status')
    if status=='DECODED': key='commands_14d'
    elif status=='DECODE_REJECTED' and decoded.get('reason')==GRIPPER_REASON:key='raw_commands_14d'
    else:return {'accepted':False,'reason':'unsupported_or_failed_decode'}
    try:values=np.asarray(decoded.get(key),dtype=np.float64)
    except (ValueError,TypeError):return {'accepted':False,'reason':'invalid_decoded_array'}
    if values.shape!=(50,14) or not np.isfinite(values).all():return {'accepted':False,'reason':'shape_or_nonfinite'}
    raw=values[0].copy();mapped=values.copy();mapped[0,[6,13]]=np.clip(raw[[6,13]],0.,1.)
    delta=mapped[0,[6,13]]-raw[[6,13]]
    candidate=copy.deepcopy(decoded);candidate[key]=mapped.tolist()
    result=admit_first_target(candidate,validate_action)
    result.update(contract='first_target_gripper_projection_v1',raw_first_target=raw.tolist(),gripper_mapping_delta=delta.tolist(),clipping_applied=bool(np.any(delta)),whole_chunk_executable=False,requires_physics_preflight=True)
    return result

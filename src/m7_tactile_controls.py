"""M7 runtime controls share exact contact-only transform with training."""
from contact_state_control import contact_state_payload
from confirmation_tactile_controls import runtime_payload as m6_payload
CONDITIONS=('zero','true','mismatch','contact_state')
WEIGHT_ARM={a:a for a in CONDITIONS}

def runtime_payload(current,condition,request_index,donor_pools):
    if condition not in CONDITIONS or type(request_index) is not int or request_index<0:
        raise ValueError('fixed M7 condition and nonnegative request required')
    if condition=='contact_state':return contact_state_payload(current),None
    return m6_payload(current,condition,request_index,donor_pools)

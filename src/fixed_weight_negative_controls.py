"""Same-weight input controls; never replace current validity, age or coverage."""
import torch
from fixed_weight_tactile_control import payload
from contact_state_control import contact_state_payload
CONDITIONS=('live','history_zero','history_mismatch','contact_bit')
def transform(current,condition,action_index,donor_pools):
    if condition not in CONDITIONS or type(action_index) is not int or action_index<0:raise ValueError('registered condition/action index required')
    if condition in ('live','history_zero'):return payload(current,condition),None
    if condition=='contact_bit':return contact_state_payload(current),None
    if not bool(current['tactile_valid'][0]):return payload(current,'live'),None
    contact=bool(((current['tactile_history'][...,:2]>0).any(-1)&(current['tactile_mask']>0)).any())
    candidates=[]
    for seed in sorted(donor_pools):
        for item in donor_pools[seed][contact]:
            meta=item['meta'];p=item['tactile']
            if meta['fold']!='train' or meta['seed']!=seed or bool(meta['contact'])!=contact:raise ValueError('training-only matching contact stratum required')
            if not bool(p['tactile_valid'][0]):raise ValueError('valid donor required')
            if p['tactile_history'].shape!=current['tactile_history'].shape:raise ValueError('donor shape mismatch')
            if torch.any((current['tactile_mask']>0)&(p['tactile_mask']<=0)):continue
            candidates.append(item)
    if not candidates:raise ValueError('no donor covering current observed bins')
    item=candidates[action_index%len(candidates)];out=payload(current,'live')
    out['tactile_history']=item['tactile']['tactile_history'].clone()
    return out,dict(item['meta'],selection_index=action_index%len(candidates),candidate_count=len(candidates))

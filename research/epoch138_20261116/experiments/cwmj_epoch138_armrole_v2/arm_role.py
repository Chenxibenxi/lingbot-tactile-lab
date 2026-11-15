import re,copy

def apply_role(command,instruction,ready_physical,validate):
 match=re.match(r'Use the (left|right) arm to ',instruction)
 if not match:raise ValueError('explicit task arm required; no simulator-state arm selection')
 side=match.group(1);inactive=list(range(7,14)) if side=='left' else list(range(7))
 held=list(ready_physical);held[6]=(held[6]+.01)/.055;held[13]=(held[13]+.01)/.055
 raw=list(command);out=list(command)
 for i in inactive:out[i]=held[i]
 checked=validate(out,gripper_units='normalized')
 return out,{'active_side':side,'source':'causal language instruction','inactive_indices':inactive,'model_command_before_role':raw,'command_after_role':out,'delta':[a-b for a,b in zip(out,raw)],'validated':checked,'scope':'task-only inactive-arm target hold; not a model-learned arm selection or physical arm freeze'}

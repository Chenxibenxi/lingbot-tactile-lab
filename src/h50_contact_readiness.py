"""Read-only contact-coverage gate for genuine H50 samples, not authorization."""
import json
from pathlib import Path
import numpy as np

def assess(samples):
    rows=[]
    for s in samples:
        x=s['input']; y=s['target']; m=s['metadata']
        h=np.asarray(x['tactile_history']);mask=np.asarray(x['tactile_mask'])
        actions=np.asarray(y['future_expert_take_action_14'])
        spans=np.asarray(y['applied_physics_spans'])
        if (h.shape!=(5,4,3) or mask.shape!=(5,4) or actions.shape!=(50,14) or spans.shape!=(50,2)
                or not all(np.isfinite(a).all() for a in (h,mask,actions,spans))
                or np.any(h<0) or np.any(h[...,2]>1) or np.any(mask<0) or np.any(mask>1)
                or spans[0,0]!=x['capture_physics_step'] or np.any(spans[:,1]<=spans[:,0])
                or np.any(spans[:-1,1]!=spans[1:,0])):
            raise ValueError('invalid causal H50 input/target contract')
        bins=(h[...,:2]>0)&(mask[...,None]>0)
        rows.append({'seed':m['seed'],'trial_id':m['trial_id'],'frame_index':m['frame_index'],
                     'capture_physics_step':x['capture_physics_step'],
                     'positive_force_feature_bins':int(bins.sum()),
                     'has_past_contact_proxy':bool(bins.any())})
    contact_seeds=sorted({r['seed'] for r in rows if r['has_past_contact_proxy']})
    return {'schema':'lingbot_h50_contact_coverage_gate_v1','rows':rows,
            'total_windows':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
            'contact_windows':sum(r['has_past_contact_proxy'] for r in rows),
            'contact_seeds':contact_seeds,'contact_signal_present':bool(contact_seeds),
            'sufficient_for_training':False,
            'reason':'coverage alone never approves training; independent validation, provenance and normalization required'}

def main():
    from h50_online_samples import OnlineH50Candidate
    reader=OnlineH50Candidate()
    report=assess(reader.sample(i,include_images=False) for i in range(len(reader)))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

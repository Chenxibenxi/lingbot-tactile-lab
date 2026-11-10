import json,hashlib
from pathlib import Path
import h5py,numpy as np
from validate_expert_data import sha256
R=Path(__file__).resolve().parents[1]
def audit_input(trial):
    trial=Path(trial);v=json.loads((trial/'result.json').read_text());m=json.loads((R/'data/reports/feedback_block_intervention_v1/manifest.json').read_text())
    assert v['adapter_sha256']==m['adapter_sha256'] and v['weight_arm']=='true'
    assert v['input_protocol_manifest_sha256']==sha256(R/'data/reports/feedback_block_intervention_v1/manifest.json')
    with h5py.File(trial/'rollout.hdf5') as f:
        for i,x in enumerate(v['requests']):
            h=f['tactile/history'][i:i+1]
            if v['arm']=='history_zero':h=np.zeros_like(h)
            assert x['input_condition']==v['arm'] and x['metadata_unchanged'] is True
            assert x['input_history_nonzero']==int(np.count_nonzero(h))
            assert x['input_history_sha256']==hashlib.sha256(h.tobytes()).hexdigest()
    return dict(same_weight_verified=True,input_history_verified=True)

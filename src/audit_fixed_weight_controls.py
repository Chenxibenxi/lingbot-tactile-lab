"""Reconstruct each runtime negative control from its source observation."""
import json,hashlib
from pathlib import Path
import h5py,torch
from validate_expert_data import sha256
from fixed_weight_negative_controls import transform
from confirmation_tactile_controls import load_frozen_training_donors
R=Path(__file__).resolve().parents[1]
def audit_input(trial):
    trial=Path(trial);v=json.loads((trial/'result.json').read_text());m=json.loads((R/'data/reports/fixed_weight_controls_v1/manifest.json').read_text())
    assert v['adapter_sha256']==m['adapter_sha256'] and v['weight_arm']=='true'
    assert v['input_protocol_manifest_sha256']==sha256(R/'data/reports/fixed_weight_controls_v1/manifest.json')
    pools,_=load_frozen_training_donors(R) if v['arm']=='history_mismatch' else ({},None)
    with h5py.File(trial/'rollout.hdf5') as f:
        for i,x in enumerate(v['requests']):
            p=dict(tactile_history=torch.tensor(f['tactile/history'][i:i+1]),tactile_mask=torch.tensor(f['tactile/coverage'][i:i+1]),tactile_age_ms=torch.tensor(f['tactile/age_ms'][i]).reshape(1),tactile_valid=torch.tensor(f['tactile/window_valid'][i]).reshape(1))
            y,donor=transform(p,v['arm'],x['executed_actions'],pools)
            assert donor==x['donor'] and x['metadata_unchanged'] is True
            assert x['input_history_sha256']==hashlib.sha256(y['tactile_history'].numpy().tobytes()).hexdigest()
            assert x['input_history_nonzero']==int(torch.count_nonzero(y['tactile_history']))
    return dict(same_weight_verified=True,donor_and_input_reconstructed=True)

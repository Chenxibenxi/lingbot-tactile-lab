"""Lazy same-run full-trajectory windows; no cross-episode actions or image cache."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms.v2 import Resize
from cwmj_preference_training_data import payload,FINGERS
from run_cwmj46_model_loop import IPC_IMAGE_TO_ENV

def causal_payload(meta):
    meta=copy.deepcopy(meta);intervals=meta['causal_tactile_intervals']
    if not 1<=len(intervals)<=3:raise ValueError('requires one to three measured intervals')
    for v in intervals:
        if not v['sensor_valid']: # invalid observations must not become NaNs or fabricated valid forces
            for f in FINGERS:v['per_finger'][f]['max_normal_N']=0.
    while len(intervals)<3:
        missing=copy.deepcopy(intervals[0]);missing['sensor_valid']=False
        for f in FINGERS:missing['per_finger'][f]['max_normal_N']=0.
        intervals.insert(0,missing)
    result=payload(meta);result['tactile_valid']=result['tactile_mask'].any(dim=1)
    result['tactile_age_ms']=torch.where(result['tactile_mask'],result['tactile_age_ms'],torch.zeros_like(result['tactile_age_ms']))
    return result

class FullTrajectoryDataset:
    def __init__(self,index_path,transform):
        self.root=Path(index_path).parent;self.index=json.loads(Path(index_path).read_text());self.rows=self.index['windows'];self.transform=transform;self.resize=Resize((256,256));self.actions={}
        for t in self.index['trajectories']:
            p=self.root/t['file']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=t['sha256']:raise ValueError('action hash mismatch')
            self.actions[t['seed']]=np.load(p,allow_pickle=False)
    def payload_only(self,i):
        r=self.rows[i];p=Path(r['observation'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['observation_sha256']:raise ValueError('observation integrity')
        return causal_payload(json.loads(p.read_text()))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];p=Path(r['observation'])
        if r['split']!='train' or hashlib.sha256(p.read_bytes()).hexdigest()!=r['observation_sha256']:raise ValueError('split or observation integrity')
        meta=json.loads(p.read_text());base={}
        with np.load(p.parent/meta['images_file'],allow_pickle=False) as images:
            for k in images.files:
                a=images[k]
                if hashlib.sha256(a.tobytes()).hexdigest()!=meta['image_sha256'][IPC_IMAGE_TO_ENV[k]]:raise ValueError('pixel hash mismatch')
                base['observation.images.'+k]=self.resize(torch.as_tensor(a).permute(2,0,1).float())
        base.update({'observation.state':torch.tensor(meta['observation']['command_history'][-1]['action_14d_normalized']),'task':meta['observation']['language']['instruction']})
        n=r['valid_steps'];action=torch.zeros(50,14);action[:n]=torch.from_numpy(self.actions[r['seed']][r['offset']:r['offset']+n].copy());pad=torch.arange(50)>=n
        x=self.transform.apply(dict(base,action=action,action_is_pad=pad),policy_eval=False)
        batch={k:x[k] if k=='image_grid_thw' else x[k].unsqueeze(0) for k in ('images','img_masks','state','lang_tokens','lang_masks','actions','joint_mask','action_is_pad','image_grid_thw')}
        batch['joint_mask']=batch['joint_mask']*(~batch['action_is_pad'].bool()).unsqueeze(-1)
        assert batch['actions'].shape==(1,50,55) and batch['joint_mask'].sum()==n*14 and not batch['joint_mask'][:,n:].any()
        return {'batch':batch,'payload':causal_payload(meta),'window':r}

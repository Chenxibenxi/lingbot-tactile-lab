"""Source-verified current RGB/contact plus censored REAL terminal actions."""
import json
from pathlib import Path
import numpy as np
from h50_online_samples import OnlineH50Candidate,INSTRUCTION
from audit_h50_batch_20261106 import BatchH50Candidate
from validate_expert_data import sha256
from paired_samples import CAMERAS

SEEDS={100023:'train',100026:'validation',100027:'validation'}

class TerminalEpisode:
    def __init__(self,seed):
        if seed not in SEEDS:raise ValueError('only frozen pilot train/validation seeds, no test')
        self.seed=seed;self.fold=SEEDS[seed]
        self.reader=OnlineH50Candidate() if seed==100023 else BatchH50Candidate(seed)
        r=self.reader
        self.anchors=tuple(i for i,f in enumerate(r.frames)
            if r.windows[f['obs_index']]['quality']['window_valid'])
        self.hashes={k:sha256(p) for k,p in r._sources.items()}
        self.trial_id=r.trial.name
    def __len__(self):return len(self.anchors)
    def sample(self,index,include_images=True):
        import h5py
        r=self.reader;i=self.anchors[index];f=r.frames[i];w=r.windows[f['obs_index']]
        for k,p in r._sources.items():
            if sha256(p)!=self.hashes[k]:raise ValueError('source bytes drifted')
        step=r.actions[i]['before_physics_step'];n=min(50,len(r.actions)-i)
        if (f['physics_step']!=step or w['anchor_physics_step']!=step or
                w['quality']['last_source_step']>step or not w['quality']['window_valid']):
            raise ValueError('noncausal current frame')
        actions=np.zeros((50,14),np.float32)
        actions[:n]=[a['command_state_14'] for a in r.actions[i:i+n]]
        valid=np.arange(50)<n
        with h5py.File(r.hdf5,'r') as h:
            state=np.asarray(h['joint_action/vector'][i],np.float32)
            if not np.allclose(state,f['command_state_14'],rtol=0,atol=1e-5):raise ValueError('state/clock mismatch')
            images={}
            if include_images:
                import cv2
                for cam in CAMERAS:
                    bgr=cv2.imdecode(np.frombuffer(h[f'observation/{cam}/rgb'][i],np.uint8),cv2.IMREAD_COLOR)
                    if bgr is None or bgr.shape!=(240,320,3):raise ValueError('invalid same-run image')
                    images[cam]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        history=np.asarray(w['features'],np.float32);mask=np.asarray(w['sample_fraction_mask'],np.float32)
        if history.shape!=(5,4,3) or not np.isfinite(history).all():raise ValueError('bad tactile window')
        return {'input':{'images_rgb':images,'command_state_14':state,'instruction':INSTRUCTION,
                'tactile_history':history,'tactile_mask':mask,'tactile_age_ms':float(w['quality']['force_age_ms']),
                'tactile_window_valid':True,'capture_physics_step':step},
                'target':{'actions_14':actions,'valid_steps':valid,'real_steps':n},
                'metadata':{'trial_id':self.trial_id,'seed':self.seed,'fold':self.fold,'frame_index':i,
                    'contact':bool(np.any(history[...,:2]>0)),'source_sha256':self.hashes}}

if __name__=='__main__':
    result=[]
    for seed in SEEDS:
        ep=TerminalEpisode(seed);samples=[ep.sample(i,False) for i in range(len(ep))]
        result.append({'seed':seed,'fold':ep.fold,'frames':len(ep),'contact_frames':sum(s['metadata']['contact'] for s in samples),
            'real_action_targets':sum(s['target']['real_steps'] for s in samples),'source_sha256':ep.hashes})
    print(json.dumps(result,indent=2))

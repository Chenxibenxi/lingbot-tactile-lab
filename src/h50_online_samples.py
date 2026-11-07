"""Read-only real H50 candidate sample builder; never fits CPU/GPU model.

One verified training-INTENT seed (s100023) yields correlated anchor windows.
Each sample has current same-run RGB/state/past tactile in input; the 50 TRUE
high-level take_action commands/variable physics spans are targets ONLY.
Approval, train-only norm and independent validation are still absent.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from align_temporal import align
from audit_online_expert_stride20 import REPO,TRIAL,CARD,audit
from paired_samples import CAMERAS
from validate_expert_data import read_jsonl,sha256

INSTRUCTION='Pick and place the two diverse bottles'


class OnlineH50Candidate:
    def __init__(self,repo=REPO):
        repo=Path(repo).resolve(strict=True)
        if repo!=REPO:raise ValueError('only pinned single-relay online H50 candidate')
        self.report=audit(repo)  # all eight source hashes, raw/HDF5 and causal clocks
        self.trial=repo/'data/paired'/TRIAL
        self.hdf5=self.trial/'demonstration/data/episode0.hdf5'
        self.actions=read_jsonl(self.trial/'online_actions.jsonl')[1:]
        raw=next(p for p in (self.trial/'force').glob('*.jsonl')
                 if not p.name.endswith(('.obs.jsonl','.frames.jsonl')))
        self._sources={'raw':raw,'obs':raw.with_suffix('.obs.jsonl'),
                       'frames':raw.with_suffix('.frames.jsonl'),'hdf5':self.hdf5,
                       'online_actions':self.trial/'online_actions.jsonl',
                       'pilot_result':self.trial/'pilot_result.json',
                       'completion':self.trial/'completion.json',
                       'success_gate':self.trial/'expert_success_gate.json'}
        self.frames=read_jsonl(raw.with_suffix('.frames.jsonl'))[1:]
        self.windows={w['observation_marker_index']:w
                      for w in align(raw,raw.with_suffix('.obs.jsonl'))}
        self.anchors=tuple(self.report['causal_h50_candidate_anchor_indices'])
        if (self.anchors!=tuple(range(2,10)) or len(self.actions)!=59 or len(self.frames)!=59):
            raise ValueError('verified online expert source changed during sample construction')

    def __len__(self):return len(self.anchors)

    def sample(self,index,*,include_images=True):
        import h5py
        if type(index) is not int or index<0 or index>=len(self):
            raise IndexError('only source-checked train-intent H50 anchors allowed')
        for name,path in self._sources.items():
            if sha256(path)!=self.report['source_file_sha256'][name]:
                raise ValueError(f'source {name} changed after verification')
        i=self.anchors[index];current=self.actions[i];frame=self.frames[i]
        step=current['before_physics_step'];window=self.windows[frame['obs_index']]
        if (frame['physics_step']!=step or current['camera_frame_index']!=i or
                window['anchor_physics_step']!=step or
                window['quality']['window_valid'] is not True or
                window['quality']['last_source_step']>step):
            raise ValueError('current RGB/causal tactile anchor changed')
        history=np.asarray(window['features'],dtype=np.float32)
        mask=np.asarray(window['sample_fraction_mask'],dtype=np.float32)
        targets=np.asarray([row['command_state_14'] for row in self.actions[i:i+50]],dtype=np.float32)
        spans=np.asarray([[row['before_physics_step'],row['after_physics_step']]
                          for row in self.actions[i:i+50]],dtype=np.int64)
        if (history.shape!=(5,4,3) or mask.shape!=(5,4) or targets.shape!=(50,14) or
                spans.shape!=(50,2) or not np.isfinite(history).all() or
                not np.isfinite(mask).all() or not np.isfinite(targets).all() or
                np.any(mask<0) or np.any(mask>1) or np.any(spans[:,1]<=spans[:,0]) or
                spans[0,0]!=step or np.any(spans[:-1,1]!=spans[1:,0])):
            raise ValueError('invalid H50 tactile/action tensor or actual controller intervals')
        images={}
        with h5py.File(self.hdf5,'r') as f:
            state=np.asarray(f['joint_action/vector'][i],dtype=np.float32)
            if include_images:
                import cv2
                for cam in CAMERAS:
                    blob=f[f'observation/{cam}/rgb'][i]
                    bgr=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
                    if bgr is None:raise ValueError('current original HDF5 RGB failed to decode')
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    if rgb.shape!=(240,320,3) or rgb.dtype!=np.uint8:
                        raise ValueError('original camera resolution or dtype changed')
                    images[cam]=rgb
        observed=np.asarray(frame['command_state_14'],dtype=np.float32)
        if (state.shape!=(14,) or not np.isfinite(state).all() or
                observed.shape!=(14,) or not np.allclose(state,observed,rtol=0,atol=1e-5)):
            raise ValueError('current HDF5 robot state no longer matches source camera marker')
        inp={'images_rgb':images,'command_state_14':state,
             'tactile_history':history,'tactile_mask':mask,
             'tactile_age_ms':np.float32(window['quality']['force_age_ms']),
             'capture_physics_step':step,'instruction':INSTRUCTION}
        return {'input':inp,
                'target':{'future_expert_take_action_14':targets,
                          'applied_physics_spans':spans},
                'metadata':{'trial_id':TRIAL,'seed':100023,'frame_index':i,
                            'task':'pick_diverse_bottles',
                            'split':'train_intent_unassigned',
                            'source_card':f'datasets/{CARD}',
                            'approved_for_training':False}}

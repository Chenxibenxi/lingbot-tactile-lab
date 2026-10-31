"""Candidate positive-bin, zero-preserving tactile normalization (TRAIN fold only).

Original all-bin median/IQR collapse to zero on this contact-sparse study.
This fitter NEVER looks at validation/test examples or approves training.
"""
from __future__ import annotations
import numpy as np


def fit_positive_contact_stats(samples, train_trial_ids, *, clip_abs=4.):
    train_ids=set(train_trial_ids)
    if not train_ids or not np.isfinite(clip_abs) or clip_abs<=0:
        raise ValueError('train trial IDs and positive finite clip bound required')
    observed_trials=set()
    full_counts=np.zeros(4,dtype=np.int64)
    positives=[[[] for _ in range(2)] for _ in range(4)]
    for sample in samples:
        trial=sample['metadata']['trial_id']
        if trial not in train_ids:continue
        x=sample['input']
        if not x['tactile_window_valid']:continue
        h=np.asarray(x['tactile_history'],dtype=np.float64)
        mask=np.asarray(x['tactile_mask'],dtype=np.float64)
        if h.shape!=(5,4,3) or mask.shape!=(5,4) or not np.isfinite(h).all() or not np.isfinite(mask).all():
            raise ValueError('invalid train tactile input/mask')
        if np.any(h<0) or np.any(h[...,2]>1) or np.any(mask<0) or np.any(mask>1):
            raise ValueError('negative or out-of-range tactile input')
        for finger in range(4):
            rows=h[:,finger][mask[:,finger]==1.]
            full_counts[finger]+=len(rows)
            for channel in range(2):
                positives[finger][channel].extend(np.log1p(rows[rows[:,channel]>0,channel]).tolist())
        observed_trials.add(trial)
    if observed_trials!=train_ids:
        raise ValueError('not every train trial has a valid tactile sample')
    median=np.zeros((4,3),dtype=float)
    scale=np.ones((4,3),dtype=float)
    counts=np.zeros((4,2),dtype=np.int64)
    for finger in range(4):
        for channel in range(2):
            values=np.asarray(positives[finger][channel],dtype=float)
            if len(values)<4:
                raise ValueError(f'finger {finger} channel {channel} has too few positive training bins')
            median[finger,channel]=np.median(values)
            q25,q75=np.percentile(values,[25,75])
            scale[finger,channel]=max(float(q75-q25),.1)
            counts[finger,channel]=len(values)
    return {'schema':'lingbot_tactile_contact_norm_v2','status':'fit_on_train_split_only',
            'training_approval':False,'source_trials':sorted(train_ids),
            'pretransform':'log1p for positive mean/peak proxy only; zero force remains zero; contact fraction unchanged',
            'median':median.tolist(),'scale_iqr_floor_0p1':scale.tolist(),
            'full_bin_counts_by_finger':full_counts.tolist(),
            'positive_force_counts_by_finger_channel':counts.tolist(),
            'positive_force_fraction_by_finger_channel':
                (counts/full_counts[:,None]).tolist(),
            'normalized_force_clip_abs':float(clip_abs)}

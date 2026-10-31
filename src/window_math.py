"""The SAME causal tactile window math for offline conversion and live inference."""
from __future__ import annotations
import math

SIDES = ('left', 'right')
FEATURES = ('mean_load_proxy_n', 'peak_load_proxy_n', 'contact_fraction')


def summarize_window(rows, anchor, links, dt, *, window_steps=25, bins=5,
                     peak_warning_n=500., marker_valid=True):
    if window_steps <= 0 or bins <= 0 or window_steps % bins:
        raise ValueError('positive window_steps divisible by bins required')
    if anchor < 0 or dt <= 0 or not math.isfinite(dt):
        raise ValueError('invalid anchor or physics timestep')
    if not math.isfinite(peak_warning_n) or peak_warning_n <= 0:
        raise ValueError('invalid impact warning threshold')
    order = [f'{side}.{name}' for side in SIDES for name in sorted(links[side])]
    by_step = {}
    for row in rows:
        step = int(row['physics_step'])
        if step > anchor or step <= 0:
            raise ValueError('future or invalid physics source step')
        if step <= anchor-window_steps:
            continue
        if step in by_step:
            raise ValueError('duplicate physics step in window')
        by_step[step] = row
    ticks_per_bin = window_steps // bins
    matrix=[];masks=[];coverage=[];maximum=0.;all_fingers=True
    for b in range(bins):
        start = anchor-window_steps+1 + b*ticks_per_bin
        ticks = [by_step.get(i) for i in range(start,start+ticks_per_bin)]
        samples = [x for x in ticks if x is not None]
        coverage.append(len(samples)/ticks_per_bin)
        values_bin=[];mask_bin=[]
        for side in SIDES:
            for finger in sorted(links[side]):
                vals=[]
                for sample in samples:
                    f = sample[side].get('fingers',{})
                    if finger not in f:
                        all_fingers=False
                        continue
                    value=float(f[finger]['load_proxy_n'])
                    if value < 0 or not math.isfinite(value):
                        raise ValueError('nonfinite or negative finger load')
                    vals.append(value)
                maximum=max(maximum,max(vals,default=0.))
                values_bin.append([sum(vals)/len(vals),max(vals),
                                   sum(v>0 for v in vals)/len(vals)] if vals else [0.,0.,0.])
                mask_bin.append(len(vals)/ticks_per_bin)
        matrix.append(values_bin);masks.append(mask_bin)
    observed=sorted(by_step)
    latest=observed[-1] if observed else None
    complete=(anchor >= window_steps and len(observed)==window_steps
              and observed[0]==anchor-window_steps+1 and observed[-1]==anchor
              and all_fingers)
    return {'shape':[bins,len(order),len(FEATURES)],'finger_order':order,
            'feature_order':list(FEATURES),'features':matrix,'sample_fraction_mask':masks,
            'quality':{'coverage':len(observed)/window_steps,'bins_coverage':coverage,
                       'last_source_step':latest,
                       'force_age_ms':(anchor-latest)*dt*1000 if latest is not None else None,
                       'per_finger_complete':all_fingers,
                       'high_impact_warning':maximum>peak_warning_n,
                       'window_valid':complete and bool(marker_valid)}}

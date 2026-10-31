"""Bounded, in-memory causal tactile history for FUTURE policy inference wiring.

No file rereads on the physics hot path. Uses window_math.summarize_window,
the exact same function used by offline paired-data alignment. This module
alone does not modify model.infer or the deployed RoboTwin client.
"""
from __future__ import annotations
from collections import deque
import math

from window_math import summarize_window


class LiveTactileWindow:
    def __init__(self, links, dt, *, window_steps=25, bins=5, max_cache_steps=125):
        if window_steps <= 0 or max_cache_steps < window_steps:
            raise ValueError('insufficient live window capacity')
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('positive timestep required')
        self.links={side:sorted(links[side]) for side in ('left','right')}
        self.dt=float(dt)
        self.window_steps=window_steps
        self.bins=bins
        self.cache=deque(maxlen=max_cache_steps)
        self.latest=0

    def push(self, physics_row):
        step=int(physics_row['physics_step'])
        if step <= self.latest:
            raise ValueError('live physics samples must be monotonically increasing')
        if not math.isclose(float(physics_row['sim_time_s']),round(step*self.dt,8),abs_tol=2e-7):
            raise ValueError('live sample clock is inconsistent')
        if not isinstance(physics_row.get('left'),dict) or not isinstance(physics_row.get('right'),dict):
            raise ValueError('missing gripper force fields')
        self.cache.append(physics_row)
        self.latest=step

    def snapshot(self, *, task, seed, episode, anchor_step=None):
        anchor=self.latest if anchor_step is None else int(anchor_step)
        if anchor != self.latest:
            raise ValueError('live inference must anchor at latest sampled physics step')
        values=summarize_window(self.cache,anchor,self.links,self.dt,
                                window_steps=self.window_steps,bins=self.bins,
                                marker_valid=bool(self.latest))
        return {'schema':'phizju_tactile_window_v1',
                'task':str(task),'seed':int(seed),'episode':int(episode),
                'anchor_physics_step':anchor,
                'anchor_sim_time_s':round(anchor*self.dt,8),
                'marker_role':'live_capture_candidate_not_automatically_model_request',
                'window_steps':self.window_steps,'dt_s':self.dt,**values}

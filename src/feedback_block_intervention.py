"""Execution-source intervention; every step still receives a fresh prediction.

This module does not change observation, RNG, adapter, or simulation clocks.
Caller must record the returned provenance separately from prediction-prefix
traces: cached actions are intentionally not the fresh prediction's prefix.
"""
import numpy as np


class BlockIntervention:
    def __init__(self, mode, block_size=10):
        if mode not in ('fresh_first', 'cached_block'):
            raise ValueError('unknown execution source')
        if type(block_size) is not int or not 1 <= block_size <= 50:
            raise ValueError('invalid block size')
        self.mode, self.block_size = mode, block_size
        self.next_action = 0
        self.cached = None
        self.anchor = None

    def select(self, prediction, *, action_index):
        if type(action_index) is not int or action_index != self.next_action:
            raise ValueError('non-contiguous action index')
        p = np.asarray(prediction)
        if p.shape != (50, 14) or not np.issubdtype(p.dtype, np.floating) or not np.isfinite(p).all():
            raise ValueError('invalid decoded prediction')
        phase = action_index % self.block_size
        if phase == 0:
            self.cached = p.copy()
            self.anchor = action_index
        source_request = action_index if self.mode == 'fresh_first' else self.anchor
        source_offset = 0 if self.mode == 'fresh_first' else phase
        command = (p[0] if self.mode == 'fresh_first' else self.cached[phase]).copy()
        self.next_action += 1
        return command, dict(execution_source=self.mode, action_index=action_index,
                            current_request_index=action_index,
                            source_request_index=source_request,
                            source_prediction_offset=source_offset,
                            block_size=self.block_size)

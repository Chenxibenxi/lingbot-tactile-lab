"""CPU-only recovery label boundaries; never certifies source provenance.

Caller must additionally validate same-run camera/force timing and completion.
No privileged geometry, phase, intervention flag or outcome is a learner input.
"""
from __future__ import annotations


def recovery_target_mask(rows, *, official_success, horizon=50):
    """Return causal segment-local target availability, not training approval.

    A query at action i may supervise only contiguous expert actions beginning
    at i. An external intervention ends that segment even if recovery succeeds.
    Short terminal tails are masked, never repeated or teacher-padded.
    """
    if type(official_success) is not bool or type(horizon) is not int or horizon < 1:
        raise ValueError('explicit boolean outcome and positive horizon required')
    prior = None
    for i, row in enumerate(rows):
        before, after = row.get('before_physics_step'), row.get('after_physics_step')
        if (row.get('action_index') != i or type(row.get('action_index')) is not int
                or type(before) is not int or type(after) is not int
                or before < 0 or after <= before or (prior is not None and before != prior)
                or type(row.get('capture_physics_step')) is not int
                or row.get('capture_physics_step') != before
                or row.get('source_role') not in ('expert', 'external_intervention')):
            raise ValueError('invalid action provenance or same-run interval')
        prior = after
    masks = []
    for i in range(len(rows)):
        valid = official_success
        mask = []
        for j in range(i, i + horizon):
            valid = valid and j < len(rows) and rows[j]['source_role'] == 'expert'
            mask.append(bool(valid))
        masks.append(mask)
    return {'target_valid_mask': masks, 'training_approval': False,
            'requires_same_run_source_and_completion_audit': True}

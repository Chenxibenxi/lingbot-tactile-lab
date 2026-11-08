"""Causal timing evidence for future cadence tests; no GPU execution."""
import math
from controlled_feedback_schedule import execution_count, matched_noise_seed


def request_timing(*, seed, executed_actions, cadence, capture_step, history_anchor,
                   window_steps, dt_s):
    count = execution_count(cadence, executed_actions)
    if (any(type(v) is not int for v in (capture_step,history_anchor,window_steps))
            or history_anchor < 0 or capture_step < history_anchor or window_steps != 25
            or type(dt_s) not in (float,int) or not math.isfinite(dt_s) or dt_s <= 0):
        raise ValueError('causal physical timing and fixed window required')
    return dict(executed_actions=executed_actions,replan_steps=cadence,
                noise_seed=matched_noise_seed(seed,executed_actions),planned_action_count=count,
                capture_physics_step=capture_step,history_anchor=history_anchor,
                requested_history_start=history_anchor-window_steps+1,
                history_age_at_capture_s=(capture_step-history_anchor)*dt_s,
                nominal_history_duration_s=window_steps*dt_s,dt_s=dt_s,
                limitation='Timing only; tactile validity remains determined by source valid mask.')


def execution_timing(request, *, physics_before, physics_after, inference_seconds):
    if (type(physics_before) is not int or type(physics_after) is not int
            or physics_before < request['capture_physics_step'] or physics_after <= physics_before
            or type(inference_seconds) not in (int,float) or not math.isfinite(inference_seconds)
            or inference_seconds < 0):
        raise ValueError('invalid action timing')
    return dict(physics_step_before=physics_before,physics_step_after=physics_after,
                history_age_at_action_start_s=(physics_before-request['history_anchor'])*request['dt_s'],
                history_age_at_action_end_s=(physics_after-request['history_anchor'])*request['dt_s'],
                inference_wall_seconds=inference_seconds,
                inference_physics_advance=physics_before-request['capture_physics_step'])

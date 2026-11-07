"""Action-indexed matched noise for a future independent feedback experiment.

Does not change the frozen M8 evaluation or authorize any GPU execution.
"""
ACTION_CAP=400
ALLOWED_REPLAN_STEPS=(1,10)


def decision_positions(replan_steps):
    if type(replan_steps) is not int or replan_steps not in ALLOWED_REPLAN_STEPS:
        raise ValueError('fixed one-action or ten-action feedback required')
    return tuple(range(0,ACTION_CAP,replan_steps))


def matched_noise_seed(environment_seed,executed_actions):
    if (type(environment_seed) is not int or not 0<=environment_seed<2**50
            or type(executed_actions) is not int or not 0<=executed_actions<ACTION_CAP):
        raise ValueError('nonnegative environment seed and actual action position required')
    return environment_seed*1000+executed_actions


def execution_count(replan_steps,executed_actions):
    positions=decision_positions(replan_steps)
    if type(executed_actions) is not int or executed_actions not in positions:
        raise ValueError('not a scheduled decision position')
    return min(replan_steps,ACTION_CAP-executed_actions)

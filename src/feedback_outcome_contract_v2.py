"""Independent terminal-outcome checks for fixed-budget feedback trials."""


def validate_outcome(result):
    if result.get('status')!='completed' or result.get('resource_censored') is not False:
        raise ValueError('uncensored completed trial required')
    actions=result.get('actions');physics=result.get('physics_steps')
    if type(actions) is not int or not 1<=actions<=400 or type(physics) is not int or not actions<=physics<=200000:
        raise ValueError('invalid actual action/physics count')
    if result.get('original_action_limit')!=400 or result.get('denoise_steps')!=10 or result.get('training_steps')!=2880:
        raise ValueError('fixed policy/action budget mismatch')
    if result.get('temporal_ensemble') is not False or result.get('training_approval') is not False:
        raise ValueError('unexpected policy mode or training approval')
    success=result.get('success');reason=result.get('termination')
    if type(success) is not bool or reason not in ('task_success','official_action_limit') or success!=(reason=='task_success'):
        raise ValueError('invalid official terminal label')
    if not success and actions!=400:
        raise ValueError('ordinary action-limit failure must actually execute all 400 actions')
    return True

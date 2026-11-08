import unittest
from feedback_outcome_contract import validate_outcome


def result():
    return dict(status='completed',resource_censored=False,actions=400,physics_steps=30000,
        original_action_limit=400,denoise_steps=10,training_steps=2880,temporal_ensemble=False,
        training_approval=False,success=False,termination='official_action_limit')


class Outcome(unittest.TestCase):
    def test_full_failure_and_early_success_allowed(self):
        self.assertTrue(validate_outcome(result()))
        x=result();x.update(success=True,termination='task_success',actions=123)
        self.assertTrue(validate_outcome(x))
    def test_short_failure_censor_budget_and_mode_rejected(self):
        for fields in ({'actions':390},{'resource_censored':True},{'physics_steps':60001},
                       {'original_action_limit':500},{'denoise_steps':2},{'training_approval':True},
                       {'success':True},{'temporal_ensemble':True}):
            with self.assertRaises(ValueError):validate_outcome(result()|fields)

import unittest
from feedback_capacity_budget import forecast


def rows():
    return [dict(seed=s,cadence=c,arm=a,requests=400//c,bytes=(400//c)*1000,wall_seconds=(400//c)*.7) for s in (100127,100128) for c in (10,1) for a in ('zero','true','mismatch','contact_state')]


class Capacity(unittest.TestCase):
    def test_full_matrix_extrapolates_both_frequencies_and_keeps_margin(self):
        x=forecast(rows());self.assertEqual(x['full_budget_bytes_estimate'],192*(400+40)*1000)
        self.assertGreater(x['required_initial_free_bytes'],x['full_budget_bytes_estimate']+x['keep_free_bytes'])
    def test_short_trial_does_not_reduce_budget(self):
        data=rows();data[0].update(requests=1,bytes=10000,wall_seconds=10.)
        self.assertGreater(forecast(data)['required_initial_free_bytes'],forecast(rows())['required_initial_free_bytes'])
    def test_incomplete_duplicate_or_empty_rejected(self):
        data=rows()
        with self.assertRaises(ValueError):forecast(data[:-1])
        data[-1]=data[0]
        with self.assertRaises(ValueError):forecast(data)
        data=rows();data[0]['requests']=0
        with self.assertRaises(ValueError):forecast(data)

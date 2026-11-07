import random
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from duration_training_contract import isolated_validation


class ContractTest(unittest.TestCase):
    def test_rng_and_mode_restored_even_on_error(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        model.train(); model[0].eval()
        model.config = SimpleNamespace(use_cache=False)
        random.seed(29); np.random.seed(30); torch.manual_seed(31)
        expected = (random.random(), np.random.random(), torch.rand(2))
        random.seed(29); np.random.seed(30); torch.manual_seed(31)
        with self.assertRaises(ValueError):
            with isolated_validation(model):
                random.random(); np.random.random(); torch.rand(99)
                model.config.use_cache = True
                raise ValueError('validation interrupted')
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        self.assertTrue(torch.equal(torch.rand(2), expected[2]))
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        self.assertFalse(model.config.use_cache)

    def test_sampling_prefix(self):
        from training_repeat_contract import balanced_order
        train = [{'meta': {'fold': 'train', 'seed': s, 'contact': c}}
                 for s in range(7) for c in (False, True) for _ in range(4)]
        for repeat in range(3):
            long = balanced_order(train, 2880, 98201+repeat)
            self.assertEqual(long[:720], balanced_order(train, 720, 98201+repeat))
            self.assertEqual(long[:1440], balanced_order(train, 1440, 98201+repeat))

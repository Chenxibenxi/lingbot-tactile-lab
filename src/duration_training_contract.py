"""Validation must not alter the continuation of an optimizer trajectory."""
from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def isolated_validation(model):
    py_state, np_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    cache = model.config.use_cache
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for module, training in modes:
            module.training = training
        model.config.use_cache = cache

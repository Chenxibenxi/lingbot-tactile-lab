import torch
from torch import nn
from contact_action_projection_lora import ActionProjectionLoRA
from action_projection_capacity import ActionProjectionCapacity

def test_zero_residual_preserves_trained_lora_and_base_frozen():
    torch.manual_seed(71)
    for full in [False,True]:
        lora=ActionProjectionLoRA(nn.Linear(16,8),rank=4,alpha=4)
        with torch.no_grad():lora.lora_B.normal_(0,.1)
        x=torch.randn(3,5,16);expected=lora(x).detach();wrapped=ActionProjectionCapacity(lora,full_residual=full)
        assert torch.equal(wrapped(x),expected)
        loss=(wrapped(x)-torch.randn_like(expected)).square().mean();loss.backward()
        assert lora.base.weight.grad is None and lora.base.bias.grad is None
        assert wrapped.capacity_bias.grad.abs().sum()>0
        if full:assert wrapped.capacity_weight.grad.abs().sum()>0
        assert lora.lora_A.grad.abs().sum()>0 and lora.lora_B.grad.abs().sum()>0

def test_fp32_linear_property_path_matches_forward():
    lora=ActionProjectionLoRA(nn.Linear(16,8),rank=4,alpha=4)
    wrapped=ActionProjectionCapacity(lora,full_residual=True)
    with torch.no_grad():wrapped.capacity_bias.fill_(.01);wrapped.capacity_weight.fill_(.02)
    x=torch.randn(2,16)
    assert torch.equal(wrapped(x),torch.nn.functional.linear(x,wrapped.weight,wrapped.bias))

from typing import Optional, Tuple

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('input_shape', default_shapes(min_dim=2, max_dim=3))
@pytest.mark.parametrize('out_dim', [16, 96, 128, 196, 384, 768, 1024])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('act_func', [None, 'sigmoid', 'tanh', 'relu', 'gelu'])
def test_linear_layers(
    input_shape: Tuple[int, ...],
    out_dim: int,
    bias: bool,
    act_func: Optional[str],
    ) -> None:
    attorch_input = create_input(input_shape)
    pytorch_input = create_input(input_shape)

    torch.manual_seed(0)
    attorch_linear = attorch.Linear(input_shape[-1], out_dim,
                                    bias=bias, act_func=act_func)

    torch.manual_seed(0)
    pytorch_linear = nn.Linear(input_shape[-1], out_dim,
                               bias=bias, device='cuda')
    pytorch_act = nn.Identity() if act_func is None else getattr(F, act_func)

    attorch_output = attorch_linear(attorch_input)
    pytorch_output = pytorch_act(pytorch_linear(pytorch_input))

    assert_close((attorch_output, pytorch_output), rtol=1e-3, atol=1e-3)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    bias_grad_pair = ((attorch_linear.bias.grad, pytorch_linear.bias.grad)
                      if bias else (torch.tensor(0), torch.tensor(0)))
    assert_close((attorch_input.grad, pytorch_input.grad),
                 (attorch_linear.weight.grad, pytorch_linear.weight.grad),
                 bias_grad_pair, rtol=1e-3, atol=1e-3)

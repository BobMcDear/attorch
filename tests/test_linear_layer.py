from typing import Optional, Tuple

import pytest
import torch
from torch import autocast, nn
from torch.nn import functional as F

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('input_shape', default_shapes(min_dim=2, max_dim=3))
@pytest.mark.parametrize('out_dim', [16, 96, 128, 196, 384, 512, 768, 1024])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('act_func', [None, 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
                                      'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu',
                                      'mish', 'softplus', 'softsign', 'tanhshrink', 'leaky_relu_0.01',
                                      'elu_1', 'celu_1', 'hardshrink_0.5', 'softshrink_0.5'])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_linear_layer(
    input_shape: Tuple[int, ...],
    out_dim: int,
    bias: bool,
    act_func: Optional[str],
    input_dtype: bool,
    amp: bool,
    subset: bool,
    ) -> None:
    if subset and (input_shape not in default_shapes(subset=True)):
        return

    if input_dtype is torch.float16 and not amp:
        return

    attorch_input = create_input(input_shape, dtype=input_dtype)
    pytorch_input = create_input(input_shape, dtype=input_dtype)

    torch.manual_seed(0)
    attorch_linear = attorch.Linear(input_shape[-1], out_dim,
                                    bias=bias,
                                    act_func=act_func)

    torch.manual_seed(0)
    pytorch_linear = nn.Linear(input_shape[-1], out_dim,
                               bias=bias, device='cuda')
    pytorch_act = nn.Identity() if act_func is None else getattr(F, act_func.rsplit('_', 1)[0])

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_linear(attorch_input)
        pytorch_output = pytorch_act(pytorch_linear(pytorch_input))

    assert_close((attorch_output, pytorch_output), rtol=1e-3, atol=1e-3)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    bias_grad_pair = ((attorch_linear.bias.grad, pytorch_linear.bias.grad)
                      if bias else (None, None))
    assert_close((attorch_input.grad, pytorch_input.grad),
                 (attorch_linear.weight.grad, pytorch_linear.weight.grad.T.contiguous()),
                 bias_grad_pair, rtol=1e-3, atol=1e-3)

from typing import Tuple

import pytest
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import init

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes(max_dim=3))
@pytest.mark.parametrize('eps', [1e-5, 1e-6])
@pytest.mark.parametrize('elementwise_affine', [False, True])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_layer_norm_layer(
    shape: Tuple[int, ...],
    eps: float,
    elementwise_affine: bool,
    bias: bool,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    attorch_input = create_input(shape, dtype=input_dtype)
    pytorch_input = create_input(shape, dtype=input_dtype)

    attorch_layer_norm = attorch.LayerNorm(shape[-1], eps, elementwise_affine, bias)
    pytorch_layer_norm = nn.LayerNorm(shape[-1], eps, elementwise_affine, bias,
                                      device='cuda')

    if elementwise_affine:
        torch.manual_seed(0)
        init.normal_(attorch_layer_norm.weight)
        if bias:
            init.normal_(attorch_layer_norm.bias)

        torch.manual_seed(0)
        init.normal_(pytorch_layer_norm.weight)
        if bias:
            init.normal_(pytorch_layer_norm.bias)

    with autocast(enabled=amp):
        attorch_output = attorch_layer_norm(attorch_input)
        pytorch_output = pytorch_layer_norm(pytorch_input)

    assert_close((attorch_output, pytorch_output),
                 rtol=1e-3, atol=1e-3)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    weight_grad_pair = ((attorch_layer_norm.weight.grad, pytorch_layer_norm.weight.grad)
                        if elementwise_affine else (None, None))
    bias_grad_pair = ((attorch_layer_norm.bias.grad, pytorch_layer_norm.bias.grad)
                      if elementwise_affine and bias else (None, None))
    assert_close((attorch_input.grad, pytorch_input.grad),
                 weight_grad_pair, bias_grad_pair,
                 rtol=1e-3, atol=1e-3)

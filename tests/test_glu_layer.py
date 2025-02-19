from typing import Tuple

import pytest
import torch
from torch import autocast
from torch.nn import functional as F

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes(max_dim=3))
@pytest.mark.parametrize('act_func', ['sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
                                      'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu',
                                      'mish', 'softplus', 'softsign', 'tanhshrink', 'leaky_relu_0.01',
                                      'elu_1', 'celu_1'])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_glu_layer(
    shape: Tuple[int, ...],
    act_func: str,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    attorch_input = create_input(shape, dtype=input_dtype)
    pytorch_input = create_input(shape, dtype=input_dtype)

    attorch_glu = attorch.GLU(act_func=act_func)
    pytorch_glu = lambda input1, input2: input1 * getattr(F, act_func.rsplit('_', 1)[0])(input2)

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_glu(attorch_input)
        pytorch_output = pytorch_glu(*pytorch_input.chunk(2, dim=-1))

    assert_close((attorch_output, pytorch_output), rtol=1e-3, atol=1e-3)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad), rtol=1e-3, atol=1e-3)

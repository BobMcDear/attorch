from typing import Tuple

import pytest
import torch
from torch import autocast, nn

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes())
@pytest.mark.parametrize('act_func', ['Sigmoid', 'LogSigmoid', 'Tanh', 'ReLU', 'GELU', 'SiLU',
                                      'ReLU6', 'Hardsigmoid', 'Hardtanh', 'Hardswish', 'SELU',
                                      'Mish', 'Softplus', 'Softsign', 'Tanhshrink', 'LeakyReLU',
                                      'ELU', 'CELU', 'Hardshrink', 'Softshrink'])
@pytest.mark.parametrize('drop_p', [0.0, 0.5])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_act_layers(
    shape: Tuple[int, ...],
    act_func: str,
    drop_p: float,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    attorch_input = create_input(shape, dtype=input_dtype)
    pytorch_input = create_input(shape, dtype=input_dtype)

    attorch_act_func = getattr(attorch, act_func)(drop_p=drop_p)
    pytorch_act_func = getattr(nn, act_func)()

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_act_func(attorch_input)
        pytorch_output = pytorch_act_func(pytorch_input)

        if drop_p > 0.0:
            mask = attorch_output == 0
            pytorch_output = torch.where(mask, 0, pytorch_output / (1 - drop_p))

    assert_close((attorch_output, pytorch_output), rtol=1e-3, atol=1e-3)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad), rtol=1e-3, atol=1e-3)

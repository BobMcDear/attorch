from typing import Tuple

import pytest
import torch
from torch import autocast, nn

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes())
@pytest.mark.parametrize('p_loss', ['SmoothL1Loss', 'L1Loss', 'MSELoss', 'HuberLoss'])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_p_loss_layers(
    shape: Tuple[int, ...],
    p_loss: str,
    reduction: str,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    attorch_input = create_input(shape, dtype=input_dtype)
    attorch_target = create_input(shape, dtype=input_dtype, seed=1)

    pytorch_input = create_input(shape, dtype=input_dtype)
    pytorch_target = create_input(shape, dtype=input_dtype, seed=1)

    attorch_loss = getattr(attorch, p_loss)(reduction=reduction)
    pytorch_loss = getattr(nn, p_loss)(reduction=reduction)

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_loss(attorch_input, attorch_target)
        pytorch_output = pytorch_loss(pytorch_input, pytorch_target)

    assert_close((attorch_output, pytorch_output), rtol=1e-3, atol=1e-3)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad),
                 (attorch_target.grad, pytorch_target.grad),
                 rtol=1e-3, atol=1e-3)

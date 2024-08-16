from typing import Tuple

import pytest
import torch
from torch import autocast, nn

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes())
@pytest.mark.parametrize('softmax', ['Softmax', 'LogSoftmax', 'Softmin'])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_softmax_layers(
    shape: Tuple[int, ...],
    softmax: str,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    attorch_input = create_input(shape, dtype=input_dtype)
    pytorch_input = create_input(shape, dtype=input_dtype)

    attorch_softmax = getattr(attorch, softmax)(dim=-1)
    pytorch_softmax = getattr(nn, softmax)(dim=-1)

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_softmax(attorch_input)
        pytorch_output = pytorch_softmax(pytorch_input)

    assert_close((attorch_output, pytorch_output))

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad))

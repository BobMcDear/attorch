from typing import Tuple

import pytest
from torch import nn

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes())
@pytest.mark.parametrize('softmax', ['Softmax', 'LogSoftmax', 'Softmin'])
def test_softmax_layers(shape: Tuple[int, ...], softmax: str) -> None:
    attorch_input = create_input(shape)
    pytorch_input = create_input(shape)

    attorch_softmax = getattr(attorch, softmax)(dim=-1)
    pytorch_softmax = getattr(nn, softmax)(dim=-1)

    attorch_output = attorch_softmax(attorch_input)
    pytorch_output = pytorch_softmax(pytorch_input)

    assert_close((attorch_output, pytorch_output))

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad))

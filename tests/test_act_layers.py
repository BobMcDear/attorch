from typing import Tuple

import pytest
from torch import nn
from torch.cuda.amp import autocast

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes())
@pytest.mark.parametrize('act_func', ['Sigmoid', 'Tanh', 'ReLU', 'GELU', 'SiLU'])
@pytest.mark.parametrize('amp', [False, True])
def test_act_layers(shape: Tuple[int, ...], act_func: str, amp: bool) -> None:
    attorch_input = create_input(shape)
    pytorch_input = create_input(shape)

    attorch_act_func = getattr(attorch, act_func)()
    pytorch_act_func = getattr(nn, act_func)()

    with autocast(enabled=amp):
        attorch_output = attorch_act_func(attorch_input)
        pytorch_output = pytorch_act_func(pytorch_input)

    assert_close((attorch_output, pytorch_output))

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad))

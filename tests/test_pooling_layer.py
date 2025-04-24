from typing import Tuple, Union

import pytest
import torch
from torch import autocast, nn

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('input_shape', default_shapes(min_dim=3, max_dim=4))
@pytest.mark.parametrize('kernel_size', [2, 3, 4])
@pytest.mark.parametrize('stride', [None, 1, 2])
@pytest.mark.parametrize('padding', [0, 1, -1])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_pooling_layer(
    input_shape: Tuple[int, ...],
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    if padding == -1:
        padding = kernel_size // 2

    attorch_input = create_input(input_shape, dtype=input_dtype)
    pytorch_input = create_input(input_shape, dtype=input_dtype)

    pooling_name = 'AvgPool2d' if len(input_shape) == 4 else 'AvgPool1d'
    attorch_pool = getattr(attorch, pooling_name)(kernel_size, stride, padding)
    pytorch_pool = getattr(nn, pooling_name)(kernel_size, stride, padding)

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_pool(attorch_input)
        pytorch_output = pytorch_pool(pytorch_input)

    assert_close((attorch_output, pytorch_output), rtol=1e-3, atol=1e-2)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad), rtol=1e-3, atol=1e-2)

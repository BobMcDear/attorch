from typing import Tuple, Union

import pytest
import torch
from torch import autocast, nn

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('input_shape', default_shapes(min_dim=3, max_dim=4))
@pytest.mark.parametrize('out_dim', [16, 96, 128, 196, 384, 512, 768, 1024])
@pytest.mark.parametrize('kernel_size', [1, 2, 3, 4, 5, 7, (5, 3), (3, 5)])
@pytest.mark.parametrize('stride', [1, 2, (1, 2), (2, 1)])
@pytest.mark.parametrize('padding', [0, 1, 3, -1])
@pytest.mark.parametrize('groups', [1, 2, 4, -1])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_conv2d_layer(
    input_shape: Tuple[int, ...],
    out_dim: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    groups: int,
    bias: bool,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    if len(input_shape) == 3 and (isinstance(kernel_size, tuple) or isinstance(padding, tuple)):
        return

    if padding == -1:
        padding = kernel_size // 2

    if groups == -1:
        groups = input_shape[1]

    if input_shape[1] % groups != 0:
        groups = 1

    conv_name = 'Conv2d' if len(input_shape) == 4 else 'Conv1d'
    attorch_input = create_input(input_shape, dtype=input_dtype)
    pytorch_input = create_input(input_shape, dtype=input_dtype)

    torch.manual_seed(0)
    attorch_conv = getattr(attorch, conv_name)(input_shape[1], out_dim,
                                               kernel_size,
                                               stride=stride, padding=padding,
                                               groups=groups, bias=bias)

    torch.manual_seed(0)
    pytorch_conv = getattr(nn, conv_name)(input_shape[1], out_dim,
                                          kernel_size,
                                          stride=stride, padding=padding,
                                          groups=groups, bias=bias,
                                          device='cuda')

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_conv(attorch_input)
        pytorch_output = pytorch_conv(pytorch_input)

    assert_close((attorch_output, pytorch_output), rtol=1e-3, atol=1e-2)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    bias_grad_pair = ((attorch_conv.bias.grad, pytorch_conv.bias.grad)
                      if bias else (torch.tensor(0), torch.tensor(0)))
    assert_close((attorch_input.grad, pytorch_input.grad),
                 (attorch_conv.weight.grad, pytorch_conv.weight.grad),
                 bias_grad_pair, rtol=1e-3, atol=1e-2)

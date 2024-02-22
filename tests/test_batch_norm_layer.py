from typing import Tuple

import pytest
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import init

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes(min_dim=2, max_dim=4))
@pytest.mark.parametrize('eps', [1e-5, 1e-6])
@pytest.mark.parametrize('momentum', [0.1, 0.2])
@pytest.mark.parametrize('affine', [False, True])
@pytest.mark.parametrize('track_running_stats', [False, True])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_batch_norm_layer(
    shape: Tuple[int, ...],
    eps: float,
    momentum: float,
    affine: bool,
    track_running_stats: bool,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if shape[0] == 1 or (input_dtype is torch.float16 and not amp):
        return

    bn_name = 'BatchNorm2d' if len(shape) == 4 else 'BatchNorm1d'
    attorch_input = create_input(shape, dtype=input_dtype)
    pytorch_input = create_input(shape, dtype=input_dtype)

    attorch_batch_norm = getattr(attorch, bn_name)(num_features=shape[1],
                                                   eps=eps, momentum=momentum,
                                                   affine=affine,
                                                   track_running_stats=track_running_stats)
    pytorch_batch_norm = getattr(nn, bn_name)(num_features=shape[1],
                                              eps=eps, momentum=momentum,
                                              affine=affine,
                                              track_running_stats=track_running_stats,
                                              device='cuda')

    if affine:
        torch.manual_seed(0)
        init.normal_(attorch_batch_norm.weight)
        init.normal_(attorch_batch_norm.bias)

        torch.manual_seed(0)
        init.normal_(pytorch_batch_norm.weight)
        init.normal_(pytorch_batch_norm.bias)

    with autocast(enabled=amp):
        attorch_output = attorch_batch_norm(attorch_input)
        pytorch_output = pytorch_batch_norm(pytorch_input)

    assert_close((attorch_output, pytorch_output),
                 (attorch_batch_norm.running_mean, pytorch_batch_norm.running_mean),
                 (attorch_batch_norm.running_var, pytorch_batch_norm.running_var))

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    weight_grad_pair = ((attorch_batch_norm.weight.grad, pytorch_batch_norm.weight.grad)
                        if affine else (None, None))
    bias_grad_pair = ((attorch_batch_norm.bias.grad, pytorch_batch_norm.bias.grad)
                      if affine else (None, None))
    assert_close((attorch_input.grad, pytorch_input.grad),
                 weight_grad_pair, bias_grad_pair,
                 rtol=1e-2, atol=1e-3)

    attorch_batch_norm.eval()
    pytorch_batch_norm.eval()

    attorch_output = attorch_batch_norm(attorch_input)
    pytorch_output = pytorch_batch_norm(pytorch_input)

    assert_close((attorch_output, pytorch_output),
                 (attorch_batch_norm.running_mean, pytorch_batch_norm.running_mean),
                 (attorch_batch_norm.running_var, pytorch_batch_norm.running_var))

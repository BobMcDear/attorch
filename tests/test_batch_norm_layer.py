from typing import Optional, Tuple

import pytest
import torch
from torch import autocast, nn
from torch.nn import functional as F, init

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes(min_dim=2, max_dim=4))
@pytest.mark.parametrize('eps', [1e-5, 1e-6])
@pytest.mark.parametrize('momentum', [0.1, 0.2])
@pytest.mark.parametrize('affine', [False, True])
@pytest.mark.parametrize('track_running_stats', [False, True])
@pytest.mark.parametrize('add_pre_act', [False, True])
@pytest.mark.parametrize('act_func', [None, 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
                                      'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu',
                                      'mish', 'leaky_relu'])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_batch_norm_layer(
    shape: Tuple[int, ...],
    eps: float,
    momentum: float,
    affine: bool,
    track_running_stats: bool,
    add_pre_act: bool,
    act_func: Optional[str],
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if shape[0] == 1 or (input_dtype is torch.float16 and not amp):
        return

    bn_name = 'BatchNorm2d' if len(shape) == 4 else 'BatchNorm1d'
    attorch_input = create_input(shape, dtype=input_dtype)
    pytorch_input = create_input(shape, dtype=input_dtype)

    if add_pre_act:
        attorch_residual = create_input(shape, dtype=input_dtype, seed=1)
        pytorch_residual = create_input(shape, dtype=input_dtype, seed=1)

    else:
        attorch_residual = pytorch_residual = None

    attorch_batch_norm = getattr(attorch, bn_name)(num_features=shape[1],
                                                   eps=eps, momentum=momentum,
                                                   affine=affine,
                                                   track_running_stats=track_running_stats,
                                                   act_func=(act_func + ('_0.01' if '_' in act_func else '')
                                                             if act_func is not None else None))
    pytorch_batch_norm = getattr(nn, bn_name)(num_features=shape[1],
                                              eps=eps, momentum=momentum,
                                              affine=affine,
                                              track_running_stats=track_running_stats,
                                              device='cuda')
    pytorch_act = nn.Identity() if act_func is None else getattr(F, act_func)

    if affine:
        torch.manual_seed(0)
        init.normal_(attorch_batch_norm.weight)
        init.normal_(attorch_batch_norm.bias)

        torch.manual_seed(0)
        init.normal_(pytorch_batch_norm.weight)
        init.normal_(pytorch_batch_norm.bias)

    with autocast('cuda', enabled=amp):
        if add_pre_act:
            attorch_output = attorch_batch_norm(attorch_input, attorch_residual)
            pytorch_output = pytorch_act(pytorch_batch_norm(pytorch_input) +
                                         pytorch_residual)

        else:
            attorch_output = attorch_batch_norm(attorch_input)
            pytorch_output = pytorch_act(pytorch_batch_norm(pytorch_input))

    assert_close((attorch_output, pytorch_output),
                 (attorch_batch_norm.running_mean, pytorch_batch_norm.running_mean),
                 (attorch_batch_norm.running_var, pytorch_batch_norm.running_var))

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    residual_grad_pair = ((attorch_residual.grad, pytorch_residual.grad)
                          if add_pre_act else (None, None))
    weight_grad_pair = ((attorch_batch_norm.weight.grad, pytorch_batch_norm.weight.grad)
                        if affine else (None, None))
    bias_grad_pair = ((attorch_batch_norm.bias.grad, pytorch_batch_norm.bias.grad)
                      if affine else (None, None))
    assert_close((attorch_input.grad, pytorch_input.grad),
                 residual_grad_pair, weight_grad_pair, bias_grad_pair,
                 rtol=1e-2, atol=1e-3)

    attorch_batch_norm.eval()
    pytorch_batch_norm.eval()

    with autocast('cuda', enabled=amp):
        if add_pre_act:
            attorch_output = attorch_batch_norm(attorch_input, attorch_residual)
            pytorch_output = pytorch_act(pytorch_batch_norm(pytorch_input) +
                                         pytorch_residual)

        else:
            attorch_output = attorch_batch_norm(attorch_input)
            pytorch_output = pytorch_act(pytorch_batch_norm(pytorch_input))

    assert_close((attorch_output, pytorch_output),
                 (attorch_batch_norm.running_mean, pytorch_batch_norm.running_mean),
                 (attorch_batch_norm.running_var, pytorch_batch_norm.running_var))

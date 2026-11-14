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
    subset: bool,
    ) -> None:
    if subset and (shape not in default_shapes(subset=True)):
        return

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


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('mixed', [False, True])
def test_l1_zero_residual_gradient(reduction, mixed):
    values = [0., -0., -2., 2., 0., -1., 1., 0.] if mixed else [0.] * 8
    x = torch.tensor(values, device='cuda', requires_grad=True)
    target = torch.zeros_like(x, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_target = target.detach().clone().requires_grad_()
    actual = attorch.L1Loss(reduction=reduction)(x, target)
    expected = nn.L1Loss(reduction=reduction)(ref_x, ref_target)
    upstream = (torch.arange(1, 9, device='cuda', dtype=x.dtype)
                if reduction == 'none' else torch.tensor(.5, device='cuda'))
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(target.grad, ref_target.grad, rtol=0, atol=0)

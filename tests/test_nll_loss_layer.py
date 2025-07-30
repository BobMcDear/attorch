from typing import Tuple

import pytest
import torch
from torch import autocast, nn

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes(min_dim=2))
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_nll_loss_layer(
    shape: Tuple[int, ...],
    reduction: str,
    weighted: bool,
    input_dtype: bool,
    amp: bool,
    subset: bool,
    ) -> None:
    if subset and (shape not in default_shapes(subset=True)):
        return

    if input_dtype is torch.float16 and not amp:
        return

    attorch_input = create_input(shape)
    pytorch_input = create_input(shape)
    target = torch.randint(0, shape[1],
                           size=(shape[0], *shape[2:]),
                           device='cuda')
    weight = (torch.randn(shape[1], device='cuda', dtype=torch.float32)
              if weighted else None)

    attorch_loss = attorch.NLLLoss(reduction=reduction, weight=weight)
    pytorch_loss = nn.NLLLoss(reduction=reduction, weight=weight)

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_loss(attorch_input, target)
        pytorch_output = pytorch_loss(pytorch_input, target)

    assert_close((attorch_output, pytorch_output))

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad))

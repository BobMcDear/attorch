from typing import Tuple

import pytest
import torch
from torch import nn
from torch.cuda.amp import autocast

import attorch
from .utils import assert_close, create_input, default_shapes


@pytest.mark.parametrize('input_shape', default_shapes(min_dim=2, max_dim=2))
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_cross_entropy_loss_layer(
    input_shape: Tuple[int, ...],
    weighted: bool,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if input_dtype is torch.float16 and not amp:
        return

    attorch_input = create_input(input_shape, dtype=input_dtype)
    pytorch_input = create_input(input_shape, dtype=input_dtype)
    target = torch.randint(0, input_shape[1],
                           size=(input_shape[0],),
                           device='cuda')
    weight = (torch.randn(input_shape[1], device='cuda')
              if weighted else None)

    attorch_loss = attorch.CrossEntropyLoss(weight=weight)
    pytorch_loss = nn.CrossEntropyLoss(weight=weight)

    with autocast(enabled=amp):
        attorch_output = attorch_loss(attorch_input, target)
        pytorch_output = pytorch_loss(pytorch_input, target)

    assert_close((attorch_output, pytorch_output), rtol=1e-3, atol=1e-3)

    attorch_output.backward()
    pytorch_output.backward()

    assert_close((attorch_input.grad, pytorch_input.grad), rtol=1e-3, atol=1e-3)

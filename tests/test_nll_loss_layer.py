from typing import Tuple

import pytest
import torch
from torch import nn

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('input_shape', default_shapes(min_dim=2))
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('weighted', [False, True])
def test_nll_loss_layer(
    input_shape: Tuple[int, ...],
    reduction: str,
    weighted: bool,
    ) -> None:
    attorch_input = create_input(input_shape)
    pytorch_input = create_input(input_shape)
    target = torch.randint(0, input_shape[1],
                           size=(input_shape[0], *input_shape[2:]),
                           device='cuda')
    weight = (torch.randn(input_shape[1], device='cuda', dtype=torch.float32)
              if weighted else None)

    attorch_loss = attorch.NLLLoss(reduction=reduction, weight=weight)
    pytorch_loss = nn.NLLLoss(reduction=reduction, weight=weight)

    attorch_output = attorch_loss(attorch_input, target)
    pytorch_output = pytorch_loss(pytorch_input, target)

    assert_close((attorch_output, pytorch_output))

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    assert_close((attorch_input.grad, pytorch_input.grad))

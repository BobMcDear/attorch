from typing import Tuple

import pytest
import torch

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes())
@pytest.mark.parametrize('drop_p', [0.0, 0.15, 0.3, 0.5, 0.75, 0.9, 1.0])
def test_dropout_layer(shape: Tuple[int, ...], drop_p: float, subset: bool) -> None:
    if subset and (shape not in default_shapes(subset=True)):
        return

    input = create_input(shape)
    dropout = attorch.Dropout(drop_p)
    output = dropout(input)
    n_zeroed = (torch.count_nonzero(input) - torch.count_nonzero(output)).item()

    if drop_p == 0:
        assert n_zeroed == 0

    elif drop_p == 1:
        assert torch.count_nonzero(output).item() == 0

    else:
        assert_close((output, torch.where(output == 0, output, input / (1 - drop_p))))
        assert_close((n_zeroed / input.numel(), drop_p), rtol=1e-1, atol=5e-2)

    output_grad = create_input_like(output)
    output.backward(output_grad)
    input_grad = torch.where(output == 0, output, output_grad / (1 - drop_p))

    assert_close((input.grad, input_grad))

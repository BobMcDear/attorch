"""
Dropout layer with PyTorch autodiff support.
"""


import warnings
from random import randint
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .dropout_kernels import dropout_backward_kernel, dropout_forward_kernel
from .types import Context


class DropoutAutoGrad(torch.autograd.Function):
    """
    Autodiff for dropout.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        drop_p: float,
        training: bool,
        ) -> Tensor:
        """
        Randomly zeroes elements in the input.

        Args:
            ctx: Context for variable storage.
            input: Input to perform dropout on.
                Can have arbitrary shape.
            drop_p: Probability of dropping an element.
            training: Flag indicating if the model is in training mode.
                If False, no dropout is applied.

        Returns:
            Input with some elements zeroed out.
        """
        ctx.do_dropout = True
        if not training or drop_p == 0:
            ctx.do_dropout = False
            return input

        ctx.drop_all = False
        if drop_p == 1:
            ctx.drop_all = True
            return torch.zeros_like(input)

        flattened_input = input.flatten()
        size = len(flattened_input)
        output = torch.empty_like(flattened_input)

        seed = randint(0, 65535)
        ctx.seed = seed
        ctx.drop_p = drop_p

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements.
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        dropout_forward_kernel[grid](flattened_input, output,
                                     size, drop_p, seed)

        return output.view_as(input)

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of dropout.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of dropout.
        """
        if not ctx.do_dropout:
            return output_grad, None, None

        if ctx.drop_all:
            return torch.zeros_like(output_grad), None, None

        orig_shape = output_grad.shape
        output_grad = output_grad.flatten()
        size = len(output_grad)
        input_grad = torch.empty_like(output_grad)

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements.
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        dropout_backward_kernel[grid](output_grad, input_grad,
                                      size, ctx.drop_p, ctx.seed)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view(orig_shape), None, None


class Dropout(nn.Dropout):
    """
    Randomly zeroes elements in the input during training.
    See also base class.

    Args:
        p: Probability of dropping an element.
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
    """
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=False)

        if inplace is True:
            warnings.warn('In-place dropout currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return DropoutAutoGrad.apply(input, self.p, self.training)

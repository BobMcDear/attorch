"""
Activation function layers with PyTorch autodiff support.
"""


import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .act_kernels import act_func_forward_kernel, act_func_backward_kernel
from .types import Context


class ActFuncAutoGrad(torch.autograd.Function):
    """
    Autodiff for activation functions.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        act_func: str,
        ) -> Tensor:
        """
        Applies an activation function to the input.

        Args:
            ctx: Context for variable storage.
            input: Input to transform.
                Can have arbitrary shape.
            act_func: Name of activation function to apply.
                Options are 'sigmoid', 'tanh', 'relu', and 'gelu'.

        Returns:
            Input transformed by the desired activation function.
        """
        flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
        flattened_input = flattened_input.flatten(0, -2)

        ctx.act_func = act_func
        ctx.flattened_shape = flattened_input.shape
        if input.requires_grad:
            ctx.save_for_backward(input)

        batch_dim, feat_dim = flattened_input.shape
        output = torch.empty_like(flattened_input,
                                  memory_format=torch.contiguous_format)

        # Launches 2D grid where each program operates over
        # one row and BLOCK_SIZE_FEAT columns.
        grid = lambda META: (batch_dim, cdiv(feat_dim, META['BLOCK_SIZE_FEAT']))
        act_func_forward_kernel[grid](flattened_input, output, feat_dim,
                                      *flattened_input.stride(), act_func)

        return output.view_as(input)

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of the activation function.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the input.

        Returns:
            Input gradient of the activation function.
        """
        (input,) = ctx.saved_tensors
        flattened_input = input.view(ctx.flattened_shape)
        output_grad = output_grad.view_as(flattened_input)

        batch_dim, feat_dim = flattened_input.shape
        input_grad = torch.empty_like(flattened_input,
                                      memory_format=torch.contiguous_format)

        # Launches 2D grid where each program operates over
        # one row and BLOCK_SIZE_FEAT columns.
        grid = lambda META: (batch_dim, cdiv(feat_dim, META['BLOCK_SIZE_FEAT']))
        act_func_backward_kernel[grid](output_grad, flattened_input, input_grad,
                                       feat_dim, *output_grad.stride(),
                                       *flattened_input.stride(), ctx.act_func)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(input), None


class Sigmoid(nn.Sigmoid):
    """
    Applies sigmoid to the input.
    See also base class.
    """
    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'sigmoid')


class Tanh(nn.Tanh):
    """
    Applies tanh to the input.
    See also base class.
    """
    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'tanh')


class ReLU(nn.ReLU):
    """
    Applies ReLU to the input.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
    """
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

        if inplace is True:
            warnings.warn('In-place ReLU currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'relu')


class GELU(nn.GELU):
    """
    Applies GELU to the input.
    See also base class.
    """
    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'gelu')

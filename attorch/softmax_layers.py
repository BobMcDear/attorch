"""
Softmax and related layers with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .softmax_kernels import softmax_backward_kernel, softmax_forward_kernel
from .types import Context


class SoftmaxAutoGrad(torch.autograd.Function):
    """
    Autodiff for softmax and related functions.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        log: bool,
        ) -> Tensor:
        """
        Normalizes the input using softmax.

        Args:
            ctx: Context for variable storage.
            input: Input to normalize.
                Can have arbitrary shape.
            log: Flag for indicating if the log of softmax should be taken.

        Returns:
            Input normalized by softmax.
        """
        flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
        flattened_input = flattened_input.flatten(0, -2)
        batch_dim, feat_dim = flattened_input.shape
        output = torch.empty_like(flattened_input)

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        softmax_forward_kernel[grid](flattened_input, output, batch_dim, feat_dim,
                                     *flattened_input.stride(), *output.stride(),
                                     log=log)

        ctx.log = log
        if input.requires_grad:
            ctx.save_for_backward(output)

        return output.view_as(input)

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of softmax.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of softmax.
        """
        (output,) = ctx.saved_tensors
        flattened_output_grad = output_grad.view_as(output)

        batch_dim, feat_dim = output.shape
        input_grad = torch.empty_like(output)

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        softmax_backward_kernel[grid](flattened_output_grad, output, input_grad,
                                      batch_dim, feat_dim,
                                      *flattened_output_grad.stride(),
                                      *output.stride(), *input_grad.stride(),
                                      log=ctx.log)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(output_grad), None


class Softmax(nn.Softmax):
    """
    Normalizes the input using softmax.
    See also base class.

    Args:
        dim: Dimension along which softmax will be computed.
            Only softmax along the last dimension is supported.
    """
    def forward(self, input: Tensor) -> Tensor:
        if self.dim != -1 and self.dim != input.ndim - 1:
            raise RuntimeError(f'Only softmax along the last dimension is supported.')

        return SoftmaxAutoGrad.apply(input, False)


class LogSoftmax(nn.LogSoftmax):
    """
    Normalizes the input using softmax and takes its log.
    See also base class.

    Args:
        dim: Dimension along which softmax will be computed.
            Only softmax along the last dimension is supported.
    """
    def forward(self, input: Tensor) -> Tensor:
        if self.dim != -1 and self.dim != input.ndim - 1:
            raise RuntimeError(f'Only softmax along the last dimension is supported.')

        return SoftmaxAutoGrad.apply(input, True)


class Softmin(nn.Softmin):
    """
    Normalizes the input using softmin.
    See also base class.

    Args:
        dim: Dimension along which softmin will be computed.
            Only softmin along the last dimension is supported.
    """
    def forward(self, input: Tensor) -> Tensor:
        if self.dim != -1 and self.dim != input.ndim - 1:
            raise RuntimeError(f'Only softmin along the last dimension is supported.')

        return SoftmaxAutoGrad.apply(-input, False)

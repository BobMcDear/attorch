"""
p-norm-induced losses with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .p_loss_kernels import p_loss_backward_kernel, p_loss_forward_kernel
from .types import Context


class PLossAutoGrad(torch.autograd.Function):
    """
    Autodiff for p-losses.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        target: Tensor,
        p_loss: int,
        reduction: str,
        ) -> Tensor:
        """
        Measures the L1 or squared L2 norm of the difference between the input
        and target (i.e., mean absolute error or mean squared error).

        Args:
            ctx: Context for variable storage.
            input: Input.
                Can have arbitrary shape.
            target: Target.
                Must be the same shape as input.
            p_loss: p-norm used to compute the error.
                Options are 1 for MAE and 2 for MSE.
            reduction: Reduction strategy for the output.
                Options are 'none' for no reduction, 'mean' for averaging the error
                across all entries, and 'sum' for summing the error across all entries.

        Returns:
            Error.
        """
        assert input.shape == target.shape, \
            f'Input shape {input.shape} and target shape {target.shape} not equal'

        flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
        flattened_input = flattened_input.flatten(0, -2)
        flattened_target = target.view_as(flattened_input)

        ctx.p_loss = p_loss
        ctx.reduction = reduction
        ctx.flattened_shape = flattened_input.shape
        if input.requires_grad or target.requires_grad:
            ctx.save_for_backward(input, target)

        batch_dim, feat_dim = flattened_input.shape
        output = (torch.empty_like(input=flattened_input,
                                   memory_format=torch.contiguous_format)
                  if reduction == 'none' else torch.tensor(0., device='cuda'))

        # Launches 2D grid where each program operates over
        # one row and BLOCK_SIZE_FEAT columns.
        grid = lambda META: (batch_dim, cdiv(feat_dim, META['BLOCK_SIZE_FEAT']))
        p_loss_forward_kernel[grid](flattened_input, flattened_target, output,
                                    batch_dim, feat_dim,
                                    *flattened_input.stride(),
                                    *flattened_target.stride(),
                                    p_loss=p_loss, reduction=reduction)

        return output.view_as(input) if reduction == 'none' else output

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of the error.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the error.
        """
        (input, target) = ctx.saved_tensors
        flattened_input = input.view(ctx.flattened_shape)
        flattened_target = target.view_as(flattened_input)

        output_grad_stride = (1, 1)
        if ctx.reduction == 'none':
            output_grad = output_grad.view_as(flattened_input)
            output_grad_stride = output_grad.stride()

        batch_dim, feat_dim = flattened_input.shape
        input_grad = torch.empty_like(flattened_input,
                                      memory_format=torch.contiguous_format)
        target_grad = torch.empty_like(input_grad)

        # Launches 2D grid where each program operates over
        # one row and BLOCK_SIZE_FEAT columns.
        grid = lambda META: (batch_dim, cdiv(feat_dim, META['BLOCK_SIZE_FEAT']))
        p_loss_backward_kernel[grid](output_grad, flattened_input, flattened_target,
                                     input_grad, target_grad,
                                     batch_dim, feat_dim,
                                     *output_grad_stride,
                                     *flattened_input.stride(),
                                     *flattened_target.stride(),
                                     p_loss=ctx.p_loss, reduction=ctx.reduction)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(input), target_grad.view_as(input), None, None


class L1Loss(nn.L1Loss):
    """
    Measures the mean absolute error between the input and target.
    See also base class.

    Args:
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
            Providing size_average and reduce overrides this argument.
        size_average: Flag for averaging instead of summing the error entries
            when reduce is True.
        reduce: Flag for averaging or summing all the error entries instead of
            returning a loss per element.
    """
    def __init__(
        self,
        reduction: str = 'mean',
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        ) -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return PLossAutoGrad.apply(input, target, 1, self.reduction)


class MSELoss(nn.MSELoss):
    """
    Measures the mean squared error between the input and target.
    See also base class.

    Args:
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
            Providing size_average and reduce overrides this argument.
        size_average: Flag for averaging instead of summing the error entries
            when reduce is True.
        reduce: Flag for averaging or summing all the error entries instead of
            returning a loss per element.
    """
    def __init__(
        self,
        reduction: str = 'mean',
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        ) -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return PLossAutoGrad.apply(input, target, 2, self.reduction)

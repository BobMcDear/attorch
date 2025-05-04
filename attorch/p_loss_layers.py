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
from .utils import get_output_dtype


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
        Measures the smooth L1, L1, or squared L2 norm of the difference between the input
        and target.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Can have arbitrary shape.
            target: Target.
                Must be the same shape as input.
            p_loss: p-norm used to compute the error.
                Options are 0 for smooth L1, 1 for L1, and 2 for squared L2.
            reduction: Reduction strategy for the output.
                Options are 'none' for no reduction, 'mean' for averaging the error
                across all entries, and 'sum' for summing the error across all entries.

        Returns:
            Error.
        """
        assert input.shape == target.shape, \
            f'Input shape {input.shape} and target shape {target.shape} not equal'

        output_dtype = get_output_dtype(input.dtype, autocast='fp32')

        ctx.p_loss = p_loss
        ctx.reduction = reduction
        ctx.output_dtype = output_dtype
        if input.requires_grad or target.requires_grad:
            ctx.save_for_backward(input, target)

        flattened_input = input.flatten()
        flattened_target = target.flatten()
        size = len(flattened_input)

        output = (torch.empty_like(flattened_input, dtype=output_dtype) if reduction == 'none'
                  else torch.empty(cdiv(size, 32), dtype=output_dtype, device=input.device))

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements.
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        p_loss_forward_kernel[grid](flattened_input, flattened_target, output,
                                    size, p_loss=p_loss, reduction=reduction)

        if reduction != 'none':
            BLOCK_SIZE = p_loss_forward_kernel.best_config.kwargs['BLOCK_SIZE']
            output = output[:cdiv(size, BLOCK_SIZE)].sum()

        else:
            output = output.view_as(input)

        return output

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
        flattened_input = input.flatten()
        flattened_target = target.flatten()
        output_grad = output_grad.flatten()

        size = len(flattened_input)
        input_grad = torch.empty_like(flattened_input, dtype=ctx.output_dtype)
        target_grad = torch.empty_like(flattened_target, dtype=ctx.output_dtype)

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements.
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        p_loss_backward_kernel[grid](output_grad, flattened_input, flattened_target,
                                     input_grad, target_grad, size,
                                     p_loss=ctx.p_loss, reduction=ctx.reduction)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(input), target_grad.view_as(input), None, None


class L1Loss(nn.L1Loss):
    """
    Measures the L1 error (mean absolute error) between the input and target.
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
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return PLossAutoGrad.apply(input, target, 1, self.reduction)


class MSELoss(nn.MSELoss):
    """
    Measures the squared L2 error (mean squared error) between the input and target.
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
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return PLossAutoGrad.apply(input, target, 2, self.reduction)


class SmoothL1Loss(nn.SmoothL1Loss):
    """
    Measures the smooth L1 error between the input and target.
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
        beta: Beta value for the softening threshold. Only 1.0 is supported.

    Raises:
        RuntimeError: A beta other than 1.0 was passed.
    """
    def __init__(
        self,
        reduction: Optional[str] = 'mean',
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        beta: float = 1.0,
        ):
        if beta != 1.0:
            raise RuntimeError('Smooth L1 only supports a beta threshold of 1.0.')

        super().__init__(size_average, reduce, reduction, beta)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return PLossAutoGrad.apply(input, target, 0, self.reduction)

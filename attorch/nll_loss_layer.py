"""
Negative log likelihood loss with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .nll_loss_kernels import nll_loss_backward_kernel, nll_loss_forward_kernel, \
    BLOCK_SIZE_BATCH_heuristic
from .types import Context


class NLLLossAutoGrad(torch.autograd.Function):
    """
    Autodiff for negative log likelihood loss.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        target: Tensor,
        reduction: str,
        weight: Optional[Tensor] = None,
        ) -> Tensor:
        """
        Measures the negative log likelihood loss between the input and target,
        with optional reweighing of each class.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Must be of shape [batch_dim, feat_dim, ...],
                where ... denotes an arbitrary number of spatial dimensions.
            target: Target.
                Must be of shape [batch_dim, ...],
                where ... denotes the same spatial dimensions as the input.
            reduction: Reduction strategy for the output.
                Options are 'none' for no reduction, 'mean' for averaging the loss
                across all entries, and 'sum' for summing the loss across all entries.
            weight: Optional class weight vector, with None for no reweighing.
                If provided, must be of shape [feat_dim].

        Returns:
            Loss.
        """
        assert len(input) == len(target) and input.shape[2:] == target.shape[1:], \
            f'Incompatible input shape ({input.shape}) and target shape ({target.shape})'
        assert weight is None or len(weight) == input.shape[1], \
            f'Dimensionality of weight vector ({len(weight)}) and input features ({input.shape[1]}) not equal'

        flattened_input = input.unsqueeze(-1) if input.ndim == 2 else input
        flattened_input = flattened_input.flatten(2, -1)

        flattened_target = target.unsqueeze(-1) if target.ndim == 1 else target
        flattened_target = flattened_target.flatten(1, -1)

        batch_dim, _, spatial_dim = flattened_input.shape
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim': batch_dim,
                                                       'spatial_dim': spatial_dim})
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH

        sum_weights = (torch.empty(out_batch_dim, dtype=input.dtype, device=input.device)
                       if reduction == 'mean' else None)
        output = (torch.empty_like(flattened_target, dtype=input.dtype)
                  if reduction == 'none' else
                  torch.empty(out_batch_dim, dtype=input.dtype, device=input.device))

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
        nll_loss_forward_kernel[grid](input, target, weight, sum_weights, output,
                                      batch_dim, spatial_dim,
                                      *flattened_input.stride(),
                                      *flattened_target.stride(),
                                      *output.stride() if reduction == 'none' else (1, 1),
                                      reduction=reduction,
                                      weighted=weight is not None)

        if reduction != 'none':
            output = output.sum()

            if reduction == 'mean' and weight is not None:
                sum_weights = sum_weights.sum()
                output /= sum_weights

        else:
            output = output.view_as(target)

        ctx.sum_weights = sum_weights
        ctx.reduction = reduction
        ctx.weight = weight
        if input.requires_grad:
            ctx.save_for_backward(input, flattened_target)

        return output

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of the loss.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the loss.
        """
        (input, flattened_target) = ctx.saved_tensors
        flattened_input = input.view(len(flattened_target), -1,
                                     flattened_target.shape[-1])
        output_grad = (output_grad.view_as(flattened_target)
                       if output_grad.ndim > 0 else output_grad)

        batch_dim, _, spatial_dim = flattened_input.shape
        input_grad = torch.zeros_like(flattened_input)

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
        nll_loss_backward_kernel[grid](output_grad, flattened_target, ctx.weight,
                                       ctx.sum_weights, input_grad,
                                       batch_dim, spatial_dim,
                                       *output_grad.stride() if ctx.reduction == 'none' else (1, 1),
                                       *flattened_target.stride(),
                                       *input_grad.stride(),
                                       reduction=ctx.reduction,
                                       weighted=ctx.weight is not None)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(input), None, None, None


class NLLLoss(nn.NLLLoss):
    """
    Measures the negative log likelihood loss between the input and target,
    with optional reweighing of each class.
    See also base class.

    Args:
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
            Providing size_average and reduce overrides this argument.
        size_average: Flag for averaging instead of summing the loss entries
            when reduce is True.
        reduce: Flag for averaging or summing all the loss entries instead of
            returning a loss per element.
        weight: Optional class weight vector, with None for no reweighing.
                If provided, must be of shape [feat_dim].
        ignore_index: This argument is not supported.
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return NLLLossAutoGrad.apply(input, target, self.reduction, self.weight)

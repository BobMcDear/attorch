"""
Cross entropy loss with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .cross_entropy_loss_kernels import cross_entropy_loss_backward_kernel, \
    cross_entropy_loss_forward_kernel
from .softmax_kernels import BLOCK_SIZE_BATCH_heuristic
from .types import Context


class CrossEntropyLossAutoGrad(torch.autograd.Function):
    """
    Autodiff for cross entropy loss.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        ) -> Tensor:
        """
        Measures the mean cross entropy loss between the input and target,
        with optional reweighing of each class.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Must be of shape [batch_dim, feat_dim].
            target: Target.
                Must be of shape [batch_dim].
            weight: Optional class weight vector, with None for no reweighing.
                If provided, must be of shape [feat_dim].

        Returns:
            Loss.
        """
        assert input.ndim == 2, f'Inputs of rank other than 2 not valid'
        assert len(input) == len(target), \
            f'Incompatible input shape ({input.shape}) and target shape ({target.shape})'
        assert weight is None or len(weight) == input.shape[1], \
            f'Dimensionality of weight vector ({len(weight)}) and input features ({input.shape[1]}) not equal'

        batch_dim, feat_dim = input.shape
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim': batch_dim,
                                                       'feat_dim': feat_dim})
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
        weighted = weight is not None
        output = torch.empty(out_batch_dim, dtype=input.dtype, device=input.device)

        if weighted:
            sum_weights = torch.empty_like(output)

        else:
            sum_weights = None

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
        cross_entropy_loss_forward_kernel[grid](input, target, weight, sum_weights, output,
                                                batch_dim, feat_dim,
                                                *input.stride(),
                                                weighted=weighted)
        output = output.sum()

        if weighted:
            sum_weights = sum_weights.sum()
            output /= sum_weights

        ctx.sum_weights = sum_weights
        ctx.weight = weight
        if input.requires_grad:
            ctx.save_for_backward(input, target)

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
                Must be a scalar.

        Returns:
            Input gradient of the loss.
        """
        (input, target) = ctx.saved_tensors
        batch_dim, feat_dim = input.shape
        input_grad = torch.empty_like(input)

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
        cross_entropy_loss_backward_kernel[grid](output_grad, target, input, ctx.weight,
                                                 ctx.sum_weights, input_grad,
                                                 batch_dim, feat_dim,
                                                 *input.stride(),
                                                 *input_grad.stride(),
                                                 weighted=ctx.weight is not None)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad, None, None


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Measures the mean cross entropy loss between the input and target,
    with optional reweighing of each class.
    See also base class.

    Note: To keep its implementation compact, this module does not support
    advanced features such as inputs with spatial dimensions or label smoothing.
    For greater flexibility, a combination of attorch.LogSoftmax and
    attorch.NLLLoss can be used.

    Args:
        reduction: Reduction strategy for the output. Only 'mean' is supported.
            Providing size_average and reduce overrides this argument.
        size_average: Flag for averaging instead of summing the loss entries
            when reduce is True. Only averaging is supported.
        reduce: Flag for averaging or summing all the loss entries instead of
            returning a loss per element. Only averaging is supported.
        weight: Optional class weight vector, with None for no reweighing.
                If provided, must be of shape [feat_dim].
        ignore_index: This argument is not supported.
        label_smoothing: This argument is not supported.

    Raises:
        RuntimeError: 1. Reduction method was not set to 'mean'.
                      2. Label smoothing is requested.
    """
    def __init__(
        self,
        reduction: str = 'mean',
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce,
                         reduction, label_smoothing)

        if self.reduction != 'mean':
            raise RuntimeError('Cross entropy only supports averaging the loss.')

        if label_smoothing > 0.0:
            raise RuntimeError('Cross entropy does not support label smoothing.')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return CrossEntropyLossAutoGrad.apply(input, target, self.weight)

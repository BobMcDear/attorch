"""
Root mean square normalization with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.amp import custom_bwd, custom_fwd
from triton import cdiv

from .rms_norm_kernels import rms_norm_backward_kernel, rms_norm_forward_kernel
from .softmax_kernels import BLOCK_SIZE_BATCH_heuristic
from .types import Context, Device


class RMSNormAutoGrad(torch.autograd.Function):
    """
    Autodiff for root mean square normalization.
    """
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Optional[Tensor] = None,
        eps: Optional[float] = None,
        ) -> Tensor:
        """
        Root-mean-square-normalizes the input.

        Args:
            ctx: Context for variable storage.
            input: Input to root-mean-square-normalize.
                Can have arbitrary shape.
            weight: Optional weights for linear transform.
                If provided, must be of shape [feat_dim].
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero. If None, it defaults to
                torch.finfo(input.dtype).eps.

        Returns:
            Root-mean-square-normalized input.
        """
        flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
        flattened_input = flattened_input.flatten(0, -2)
        batch_dim, feat_dim = flattened_input.shape
        eps = torch.finfo(input.dtype).eps if eps is None else eps

        output = torch.empty_like(flattened_input)

        scale_by_weight = weight is not None
        requires_grad = (input.requires_grad or
                         (scale_by_weight and weight.requires_grad))

        if requires_grad:
            inv_rms = torch.empty(batch_dim,
                                  device=input.device,
                                  dtype=torch.float32)

        else:
            inv_rms = None

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        rms_norm_forward_kernel[grid](flattened_input, weight,
                                      inv_rms, output,
                                      batch_dim, feat_dim,
                                      *flattened_input.stride(), *output.stride(),
                                      eps,
                                      scale_by_weight=scale_by_weight,
                                      save_stats=requires_grad)

        ctx.scale_by_weight = scale_by_weight
        if requires_grad:
            ctx.save_for_backward(flattened_input, inv_rms, weight)

        return output.view_as(input)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of root mean square normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of root mean square normalization.
        """
        scale_by_weight = ctx.scale_by_weight
        (flattened_input, inv_rms, weight) = ctx.saved_tensors
        flattened_output_grad = output_grad.view_as(flattened_input)

        batch_dim, feat_dim = flattened_output_grad.shape
        input_grad = torch.empty_like(flattened_output_grad)

        if scale_by_weight:
            BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim': batch_dim,
                                                           'feat_dim': feat_dim})
            out_batch_dim = batch_dim // BLOCK_SIZE_BATCH

            weight_grad = torch.empty((out_batch_dim, feat_dim),
                                      device=flattened_input.device)

        else:
            weight_grad = None

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        rms_norm_backward_kernel[grid](flattened_output_grad, flattened_input,
                                       inv_rms, weight,
                                       input_grad, weight_grad,
                                       batch_dim, feat_dim,
                                       *flattened_output_grad.stride(),
                                       *flattened_input.stride(),
                                       *input_grad.stride(),
                                       *weight_grad.stride() if scale_by_weight else (1, 1),
                                       scale_by_weight=scale_by_weight)

        if scale_by_weight:
            weight_grad = weight_grad.sum(dim=0)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(output_grad), weight_grad, None


class RMSNorm(nn.RMSNorm):
    """
    Root-mean-square-normalizes the input.
    See also base class.

    Args:
        normalized_shape: Dimensionality of last feature that is normalized.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero. If None, it defaults to
            torch.finfo(input.dtype).eps.
        elementwise_affine: Flag for scaling the normalized output by weights.
        device: Device to use.
        dtype: Dtype of layer.

    Raises:
        RuntimeError: Normalized shape was not an integer.
    """
    def __init__(
        self,
        normalized_shape: int,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        if not isinstance(normalized_shape, int):
            raise RuntimeError('Normalized shape must be an integer.')

        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        return RMSNormAutoGrad.apply(input, self.weight, self.eps)

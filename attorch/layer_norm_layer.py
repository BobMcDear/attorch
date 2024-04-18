"""
Layer normalization with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .layer_norm_kernels import layer_norm_backward_kernel, layer_norm_forward_kernel
from .softmax_kernels import BLOCK_SIZE_BATCH_heuristic
from .types import Context, Device
from .utils import get_output_dtype


class LayerNormAutoGrad(torch.autograd.Function):
    """
    Autodiff for layer normalization.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
        autocast_to_fp32: bool = True,
        ) -> Tensor:
        """
        Layer-normalizes the input.

        Args:
            ctx: Context for variable storage.
            input: Input to layer-normalize.
                Can have arbitrary shape.
            weight: Optional weights for affine transform.
                If provided, must be of shape [feat_dim].
            bias: Optional bias vector for affine transform when weight is provided.
                If provided, must be of shape [feat_dim].
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero.
            autocast_to_fp32: Flag for autocasting the output dtype to fp32.
                If False, the input dtype flows through.

        Returns:
            Layer-normalized input.
        """
        flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
        flattened_input = flattened_input.flatten(0, -2)
        batch_dim, feat_dim = flattened_input.shape

        output_dtype = get_output_dtype(input.dtype,
                                        autocast='fp32' if autocast_to_fp32 else None)
        output = torch.empty_like(flattened_input, dtype=output_dtype)

        scale_by_weight = weight is not None
        add_bias = scale_by_weight and bias is not None
        requires_grad = (input.requires_grad or
                         (scale_by_weight and weight.requires_grad) or
                         (add_bias and bias.requires_grad))

        if requires_grad:
            mean = torch.empty(batch_dim,
                               device=input.device,
                               dtype=torch.float32)
            inv_std = torch.empty(batch_dim,
                                  device=input.device,
                                  dtype=torch.float32)

        else:
            mean = inv_std = None

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        layer_norm_forward_kernel[grid](flattened_input, weight, bias,
                                        mean, inv_std, output,
                                        batch_dim, feat_dim,
                                        *flattened_input.stride(), *output.stride(),
                                        eps,
                                        scale_by_weight=scale_by_weight,
                                        add_bias=add_bias,
                                        save_stats=requires_grad)

        ctx.scale_by_weight = scale_by_weight
        ctx.add_bias = add_bias
        ctx.output_dtype = output_dtype
        if requires_grad:
            ctx.save_for_backward(flattened_input, mean, inv_std, weight)

        return output.view_as(input)

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of layer normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of layer normalization.
        """
        scale_by_weight, add_bias = ctx.scale_by_weight, ctx.add_bias
        (flattened_input, mean, inv_std, weight) = ctx.saved_tensors
        flattened_output_grad = output_grad.view_as(flattened_input)

        batch_dim, feat_dim = flattened_output_grad.shape
        input_grad = torch.empty_like(flattened_output_grad,
                                      dtype=ctx.output_dtype)

        if scale_by_weight:
            BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim': batch_dim,
                                                           'feat_dim': feat_dim})
            out_batch_dim = batch_dim // BLOCK_SIZE_BATCH

            weight_grad = torch.empty((out_batch_dim, feat_dim),
                                      device=flattened_input.device)
            if add_bias:
                bias_grad = torch.empty((out_batch_dim, feat_dim),
                                        device=flattened_input.device)

            else:
                bias_grad = None

        else:
            weight_grad = bias_grad = None

        # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        layer_norm_backward_kernel[grid](flattened_output_grad, flattened_input,
                                         mean, inv_std, weight,
                                         input_grad, weight_grad, bias_grad,
                                         batch_dim, feat_dim,
                                         *flattened_output_grad.stride(),
                                         *flattened_input.stride(),
                                         *input_grad.stride(),
                                         *weight_grad.stride() if scale_by_weight else (1, 1),
                                         *bias_grad.stride() if scale_by_weight and add_bias else (1, 1),
                                         scale_by_weight=scale_by_weight,
                                         add_bias=add_bias)

        if scale_by_weight:
            weight_grad = weight_grad.sum(dim=0)
            if add_bias:
                bias_grad = bias_grad.sum(dim=0)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(output_grad), weight_grad, bias_grad, None


class LayerNorm(nn.LayerNorm):
    """
    Layer-normalizes the input.
    See also base class.

    Note: During automatic mixed precision training, PyTorch autocasts
    the output dtype of layer normalization to fp32. This conversion is not necessary
    and retaining the input dtype is generally numerically stable in addition to
    being slightly more efficient. Layer normalization in attorch thus offers
    an additional argument, autocast_to_fp32, to manually disable this
    autocasting behaviour.

    Args:
        normalized_shape: Dimensionality of last feature that is normalized.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        elementwise_affine: Flag for scaling the normalized output by weights.
        bias: Flag for adding a bias vector to the normalized output
            if elementwise_affine is True.
        autocast_to_fp32: Flag for autocasting the output dtype to fp32.
            If False, the input dtype flows through.
        device: Device to use.
        dtype: Dtype of layer.

    Raises:
        RuntimeError: 1. Normalized shape was not an integer.
    """
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        autocast_to_fp32: bool = True,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        if not isinstance(normalized_shape, int):
            raise RuntimeError('Normalized shape must be an integer.')

        super().__init__(normalized_shape, eps, elementwise_affine, bias,
                         device, dtype)
        self.autocast_to_fp32 = autocast_to_fp32

    def forward(self, input: Tensor) -> Tensor:
        return LayerNormAutoGrad.apply(input, self.weight, self.bias, self.eps,
                                       self.autocast_to_fp32)

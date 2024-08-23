"""
Kernels for root mean square normalization.
"""


import triton
import triton.language as tl
from triton import next_power_of_2

from .softmax_kernels import BLOCK_SIZE_BATCH_heuristic
from .utils import warps_kernel_configs


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'feat_dim'],
)
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
                    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def rms_norm_forward_kernel(
    input_pointer, weight_pointer,
    inv_rms_pointer, output_pointer,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    output_batch_stride, output_feat_stride,
    eps,
    scale_by_weight: tl.constexpr, save_stats: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Root-mean-square-normalizes the input.

    Args:
        input_pointer: Pointer to the input to root-mean-square-normalize.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to optional weights for linear transform.
            The weights, if provided, must be of shape [feat_dim].
        inv_rms_pointer: Pointer to an optional container the input's inverse
            root mean square is written to if save_stats is True.
            The container, if provided, must be of shape [batch_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        scale_by_weight: Flag for scaling the normalized output by weights.
        save_stats: Flag for saving the root mean square.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes BLOCK_SIZE_BATCH rows and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    input_pointer += (input_batch_stride * batch_offset[:, None] +
                      input_feat_stride * feat_offset[None, :])
    output_pointer += (output_batch_stride * batch_offset[:, None] +
                       output_feat_stride * feat_offset[None, :])

    input = tl.load(input_pointer,
                    mask=batch_mask[:, None] & feat_mask[None, :]).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(input * input, axis=1) / feat_dim + eps)
    output = input * inv_rms[:, None]

    if save_stats:
        tl.store(inv_rms_pointer + batch_offset, inv_rms, mask=batch_mask)

    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        output *= weight

    tl.store(output_pointer, output,
             mask=batch_mask[:, None] & feat_mask[None, :])


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'feat_dim'],
)
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
                    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def rms_norm_backward_kernel(
    output_grad_pointer, input_pointer, inv_rms_pointer, weight_pointer,
    input_grad_pointer, weight_grad_pointer,
    batch_dim, feat_dim,
    output_grad_batch_stride, output_grad_feat_stride,
    input_batch_stride, input_feat_stride,
    input_grad_batch_stride, input_grad_feat_stride,
    weight_grad_batch_stride, weight_grad_feat_stride,
    scale_by_weight: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Calculates the input gradient of root mean square normalization.

    Args:
        output_grad_pointer: Pointer to root mean square normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        inv_rms_pointer: Pointer to the input's inverse root mean square.
            The inverse root mean square should be of shape [batch_dim].
        weight_pointer: Pointer to optional weights if affine transform occurred.
            The weights, if provided, must be of shape [feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        weight_grad_pointer: Pointer to an optional container the weights' row-wise gradients
            are written to if scale_by_weight is True, which should later be summed.
            The container, if provided, must be of shape [batch_dim/BLOCK_SIZE_BATCH, feat_dim].
        bias_grad_pointer: Pointer to an optional container the bias vector's row-wise gradients
            are written to if scale_by_weight and add_bias are True, which should later be summed.
            The container, if provided, must be of shape [batch_dim/BLOCK_SIZE_BATCH, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        weight_grad_batch_stride: Stride necessary to jump one element along the
            weight gradient container's batch dimension.
        weight_grad_feat_stride: Stride necessary to jump one element along the
            weight gradient container's feature dimension.
        scale_by_weight: Flag for scaling the normalized output by weights.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes a single row and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    output_grad_pointer += (output_grad_batch_stride * batch_offset[:, None] +
                            output_grad_feat_stride * feat_offset[None, :])
    input_pointer += (input_batch_stride * batch_offset[:, None] +
                      input_feat_stride * feat_offset[None, :])
    input_grad_pointer += (input_grad_batch_stride * batch_offset[:, None] +
                           input_grad_feat_stride * feat_offset[None, :])

    output_grad = tl.load(output_grad_pointer,
                          mask=batch_mask[:, None] & feat_mask[None, :]).to(tl.float32)
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :]).to(tl.float32)
    inv_rms = tl.load(inv_rms_pointer + batch_offset, mask=batch_mask)
    pre_lin = input * inv_rms[:, None]

    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        weight_output_grad_prod = weight * output_grad

    else:
        weight_output_grad_prod = output_grad

    term1 = input * tl.sum(input * weight_output_grad_prod, axis=1)
    term2 = inv_rms[:, None] * inv_rms[:, None]
    input_grad = (inv_rms[:, None] *
                  (weight_output_grad_prod - term1 * term2 / feat_dim))

    tl.store(input_grad_pointer, input_grad,
             mask=batch_mask[:, None] & feat_mask[None, :])

    if scale_by_weight:
        weight_grad_pointer += (weight_grad_batch_stride * batch_pid +
                                weight_grad_feat_stride * feat_offset)
        tl.store(weight_grad_pointer,
                 tl.sum(output_grad * pre_lin, axis=0),
                 mask=feat_mask)

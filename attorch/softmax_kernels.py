"""
Kernels for softmax and related functions.
"""


from typing import Dict

import triton
import triton.language as tl
from triton import next_power_of_2

from .utils import warps_kernel_configs


def BLOCK_SIZE_BATCH_heuristic(args: Dict) -> int:
    """
    Approximates an appropriate batch block size for softmax using a heuristic.

    Args:
        args: Arguments to softmax kernel.

    Returns:
        Appropriate batch block size.
    """
    # This heuristic was derived manually.
    # Essentially, if the batch dimension is greater than 1024,
    # for small feature sizes (less than 64), it is much more efficient
    # to process multiple rows at once in a given program.
    # Specifically, each time the number of samples is doubled,
    # the block size across the batch dimension should be doubled too,
    # with an upper bound of 128.
    return (min(max(1, next_power_of_2(args['batch_dim'] // 2 ** 10)), 128)
            if args['feat_dim'] < 64 else 1)


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'feat_dim'],
)
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
                    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def softmax_forward_kernel(
    input_pointer, output_pointer,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    output_batch_stride, output_feat_stride,
    log: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Normalizes the input using softmax.

    Args:
        input_pointer: Pointer to the input to normalize.
            The input must be of shape [batch_dim, feat_dim].
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
        log: Flag for indicating if the log of softmax should be taken.
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

    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :],
                    other=-float('inf'))
    input -= tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=1)[:, None]

    if log:
        output = input - tl.log(denominator)

    else:
        output = numerator / denominator

    tl.store(output_pointer, output, mask=batch_mask[:, None] & feat_mask[None, :])


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'feat_dim'],
)
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
                    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def softmax_backward_kernel(
    output_grad_pointer, output_pointer, input_grad_pointer,
    batch_dim, feat_dim,
    output_grad_batch_stride, output_grad_feat_stride,
    output_batch_stride, output_feat_stride,
    input_grad_batch_stride, input_grad_feat_stride,
    log: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Calculates the input gradient of softmax.

    Args:
        output_grad_pointer: Pointer to softmax's output gradients.
            The output container must be of shape [batch_dim, feat_dim].
        output_pointer: Pointer to softmax's output.
            The output must be of shape [batch_dim, feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        log: Flag indicating if log of softmax was taken.
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
    output_pointer += (output_batch_stride * batch_offset[:, None] +
                       output_feat_stride * feat_offset[None, :])
    input_grad_pointer += (input_grad_batch_stride * batch_offset[:, None] +
                           input_grad_feat_stride * feat_offset[None, :])

    output_grad = tl.load(output_grad_pointer,
                          mask=batch_mask[:, None] & feat_mask[None, :])
    output = tl.load(output_pointer,
                     mask=batch_mask[:, None] & feat_mask[None, :])

    if log:
        input_grad = (output_grad -
                      tl.exp(output) * tl.sum(output_grad, axis=1)[:, None])

    else:
        input_grad = output * (output_grad -
                               tl.sum(output_grad * output, axis=1)[:, None])

    tl.store(input_grad_pointer, input_grad,
             mask=batch_mask[:, None] & feat_mask[None, :])

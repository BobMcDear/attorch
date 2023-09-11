"""
Kernels for softmax and related functions.
"""


import triton
import triton.language as tl
from triton import next_power_of_2


@triton.autotune(
    configs=[triton.Config({}, num_warps=2**i) for i in range(6)],
    key=['feat_dim'],
)
@triton.heuristics({'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def softmax_forward_kernel(
    input_pointer, output_pointer,
    feat_dim,
    input_batch_stride, input_feat_stride,
    log: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Normalizes the input using softmax.

    Args:
        input_pointer: Pointer to the input to normalize.
            The input must be of shape [batch_dim, feat_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim] and contiguous.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        log: Flag for indicating if the log of softmax should be taken.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes a single row and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)

    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < feat_dim

    input_pointer += (batch_pid * input_batch_stride +
                      feat_offset * input_feat_stride)
    output_pointer += batch_pid * feat_dim + feat_offset

    input = tl.load(input_pointer, mask=feat_mask, other=-float('inf'))
    input -= tl.max(input, axis=0)
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=0)

    if log:
        output = input - tl.log(denominator)

    else:
        output = numerator / denominator

    tl.store(output_pointer, output, mask=feat_mask)


@triton.autotune(
    configs=[triton.Config({}, num_warps=2**i) for i in range(6)],
    key=['feat_dim'],
)
@triton.heuristics({'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def softmax_backward_kernel(
    output_grad_pointer, output_pointer, input_grad_pointer,
    feat_dim,
    output_grad_batch_stride, output_grad_feat_stride,
    log: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Calculates the input gradient of softmax.

    Args:
        output_grad_pointer: Pointer to softmax's output gradients.
            The output container must be of shape [batch_dim, feat_dim].
        output_pointer: Pointer to softmax's output.
            The output must be of shape [batch_dim, feat_dim] and contiguous.
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim] and contiguous.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        log: Flag indicating if log of softmax was taken.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes a single row and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)

    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < feat_dim

    output_grad_pointer += (batch_pid * output_grad_batch_stride +
                            feat_offset * output_grad_feat_stride)
    output_pointer += batch_pid * feat_dim + feat_offset
    input_grad_pointer += batch_pid * feat_dim + feat_offset

    output_grad = tl.load(output_grad_pointer, mask=feat_mask)
    output = tl.load(output_pointer, mask=feat_mask)

    if log:
        input_grad = output_grad - tl.exp(output) * tl.sum(output_grad)

    else:
        input_grad = output * (output_grad - tl.sum(output_grad * output))

    tl.store(input_grad_pointer, input_grad, mask=feat_mask)

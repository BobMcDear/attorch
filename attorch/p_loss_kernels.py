"""
Kernels for p-norm-induced losses.
"""


import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_FEAT': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_FEAT': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_FEAT': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_FEAT': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_FEAT': 1024}, num_warps=4),
    ],
    key=['feat_dim'],
    reset_to_zero=['output_pointer'],
)
@triton.jit
def p_loss_forward_kernel(
    input_pointer, target_pointer, output_pointer,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    target_batch_stride, target_feat_stride,
    p_loss: tl.constexpr, reduction: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Measures the L1 or squared L2 norm of the difference between the input
    and target (i.e., mean absolute error or mean squared error).

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, feat_dim].
        output_pointer: Pointer to a container the error is written to.
            The container must be of shape [batch_dim, feat_dim] and contiguous
            if reduction is 'none', and otherwise a scalar.
        batch_dim: Batch dimension of the input and target.
        feat_dim: Dimensionality of the features of the input and target.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_feat_stride: Stride necessary to jump one element along the
            target's feature dimension.
        p_loss: p-norm used to compute the error.
            Options are 1 for MAE and 2 for MSE.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes a single row and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)
    feat_pid = tl.program_id(axis=1)

    feat_offset = feat_pid * BLOCK_SIZE_FEAT + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < feat_dim

    input_pointer += (batch_pid * input_batch_stride +
                      feat_offset * input_feat_stride)
    target_pointer += (batch_pid * target_batch_stride +
                       feat_offset * target_feat_stride)

    input = tl.load(input_pointer, mask=feat_mask)
    target = tl.load(target_pointer, mask=feat_mask)
    diff = input - target

    if p_loss == 1:
        error = tl.abs(diff)

    elif p_loss == 2:
        error = diff * diff

    if reduction == 'none':
        output_pointer += batch_pid * feat_dim + feat_offset
        tl.store(output_pointer, error, mask=feat_mask)

    elif reduction == 'mean':
        tl.atomic_add(output_pointer, tl.sum(error) / (batch_dim * feat_dim))

    elif reduction == 'sum':
        tl.atomic_add(output_pointer, tl.sum(error))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_FEAT': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_FEAT': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_FEAT': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_FEAT': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_FEAT': 1024}, num_warps=4),
    ],
    key=['feat_dim'],
)
@triton.jit
def p_loss_backward_kernel(
    output_grad_pointer, input_pointer, target_pointer,
    input_grad_pointer, target_grad_pointer,
    batch_dim, feat_dim,
    output_grad_batch_stride, output_grad_feat_stride,
    input_batch_stride, input_feat_stride,
    target_batch_stride, target_feat_stride,
    p_loss: tl.constexpr, reduction: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Calculates the input gradient of the mean absolute error or
    mean squared error.

    Args:
        output_grad_pointer: Pointer to the error's output gradients.
            The output container must be a scalar or of shape [batch_dim, feat_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim] and contiguous.
        target_grad_pointer: Pointer to a container the target's gradients are written to.
            The container must be of shape [batch_dim, feat_dim] and contiguous.
        batch_dim: Batch dimension of the input and target.
        feat_dim: Dimensionality of the features of the input and target.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension if it is not a scalar.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension if it is not a scalar.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_feat_stride: Stride necessary to jump one element along the
            target's feature dimension.
        p_loss: p-norm used to compute the error whose gradient is calculated.
            Options are 1 for MAE and 2 for MSE.
        reduction: Reduction strategy for the output whose gradient is calculated.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes a single row and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)
    feat_pid = tl.program_id(axis=1)

    feat_offset = feat_pid * BLOCK_SIZE_FEAT + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < feat_dim

    input_pointer += (batch_pid * input_batch_stride +
                      feat_offset * input_feat_stride)
    target_pointer += (batch_pid * target_batch_stride +
                       feat_offset * target_feat_stride)

    output_grad_mask = None
    if reduction == 'none':
        output_grad_pointer += (batch_pid * output_grad_batch_stride +
                                feat_offset * output_grad_feat_stride)
        output_grad_mask = feat_mask

    input = tl.load(input_pointer, mask=feat_mask)
    target = tl.load(target_pointer, mask=feat_mask)
    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask)

    if p_loss == 1:
        input_grad = tl.where(target <= input, 1, -1)

    elif p_loss == 2:
        input_grad = 2 * (input - target)

    if reduction == 'mean':
        input_grad /= batch_dim * feat_dim

    input_grad *= output_grad
    input_grad_pointer += batch_pid * feat_dim + feat_offset
    target_grad_pointer += batch_pid * feat_dim + feat_offset

    tl.store(input_grad_pointer, input_grad, mask=feat_mask)
    tl.store(target_grad_pointer, -input_grad, mask=feat_mask)

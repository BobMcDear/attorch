"""
Kernels for negative log likelihood loss.
"""


import triton
import triton.language as tl
from triton import next_power_of_2

from .utils import warps_kernel_configs


def BLOCK_SIZE_BATCH_heuristic(args) -> int:
    """
    Approximates an appropriate batch block size for NLL loss using a heuristic.

    Args:
        args: Arguments to NLL loss kernel.

    Returns:
        Appropriate batch block size.
    """
    # This heuristic was derived manually.
    # Essentially, if the batch dimension is greater than 1024,
    # for small spatial sizes (less than 64), it is much more efficient
    # to process multiple rows at once in a given program.
    # Specifically, each time the number of samples is doubled,
    # the block size across the batch dimension should be doubled too,
    # with an upper bound of 128.
    return (min(max(1, next_power_of_2(args['batch_dim'] // 2 ** 10)), 128)
            if args['spatial_dim'] < 64 else 1)


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'spatial_dim'],
)
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
                    'BLOCK_SIZE_SPATIAL': lambda args: next_power_of_2(args['spatial_dim'])})
@triton.jit
def nll_loss_forward_kernel(
    input_pointer, target_pointer, weight_pointer,
    sum_weights_pointer, output_pointer,
    batch_dim, spatial_dim,
    input_batch_stride, input_feat_stride, input_spatial_stride,
    target_batch_stride, target_spatial_stride,
    output_batch_stride, output_spatial_stride,
    reduction: tl.constexpr, weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_SPATIAL: tl.constexpr,
    ):
    """
    Measures the negative log likelihood between the input and target,
    with optional reweighing of each class.

    Args:
        input_pointer: Pointer to the input normalize.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, spatial_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to a container the sum of the class weights is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        output_pointer: Pointer to a container the loss is written to.
            The container must be of shape [batch_dim, spatial_dim] if reduction is 'none',
            and otherwise of shape [batch_dim/BLOCK_SIZE].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_spatial_stride: Stride necessary to jump one element along the
            target's spatial dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_spatial_stride: Stride necessary to jump one element along the
            output container's spatial dimension.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
            If a reduction method is specified, the reduced result of each
            program is written to a separate index in the summed weights and
            output container, which should later be summed.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    # This program processes BLOCK_SIZE_BATCH rows and
    # BLOCK_SIZE_SPATIAL spatial elements.
    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)

    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim

    target_pointer += (target_batch_stride * batch_offset[:, None] +
                       target_spatial_stride * spatial_offset[None, :])
    target = tl.load(target_pointer,
                     mask=batch_mask[:, None] & spatial_mask[None, :])

    input_pointer += (input_feat_stride * target +
                      input_batch_stride * batch_offset[:, None] +
                      input_spatial_stride * spatial_offset[None, :])
    input = tl.load(input_pointer,
                    mask=batch_mask[:, None] & spatial_mask[None, :])

    output = -input
    if weighted:
        weight = tl.load(weight_pointer + target,
                         mask=batch_mask[:, None] & spatial_mask[None, :])
        output *= weight

    if reduction == 'none':
        output_pointer += (output_batch_stride * batch_offset[:, None] +
                           output_spatial_stride * spatial_offset[None, :])
        tl.store(output_pointer, output,
                 mask=batch_mask[:, None] & spatial_mask[None, :])

    elif reduction == 'mean':
        if weighted:
            tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))
            tl.store(output_pointer + batch_pid, tl.sum(output))

        else:
            tl.store(output_pointer + batch_pid,
                    tl.sum(output) / (batch_dim * spatial_dim))

    elif reduction == 'sum':
        tl.store(output_pointer + batch_pid, tl.sum(output))


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'spatial_dim'],
)
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
                    'BLOCK_SIZE_SPATIAL': lambda args: next_power_of_2(args['spatial_dim'])})
@triton.jit
def nll_loss_backward_kernel(
    output_grad_pointer, target_pointer, weight_pointer,
    sum_weights_pointer, input_grad_pointer,
    batch_dim, spatial_dim,
    output_grad_batch_stride, output_grad_feat_stride,
    target_batch_stride, target_spatial_stride,
    input_grad_batch_stride, input_grad_feat_stride, input_grad_spatial_stride,
    reduction: tl.constexpr, weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_SPATIAL: tl.constexpr,
    ):
    """
    Calculates the input gradient of negative log likelihood loss.

    Args:
        output_grad_pointer: Pointer to the loss's output gradients.
            The output container must be of shape [batch_dim, spatial_dim]
            if reduction is 'none', and otherwise [batch_dim/BLOCK_SIZE_BATCH].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, spatial_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to the sum of the class weights if the classes were weighed.
            The sum of weights must be a scalar.
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim] and zeroed.
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_spatial_stride: Stride necessary to jump one element along the
            target's spatial dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        input_grad_spatial_stride: Stride necessary to jump one element along the
            input gradient container's spatial dimension.
        reduction: Reduction strategy for the output whose gradient is calculated.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    # This program processes BLOCK_SIZE_BATCH rows and
    # BLOCK_SIZE_SPATIAL spatial elements.
    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)

    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim

    output_grad_mask = None
    if reduction == 'none':
        output_grad_pointer += (output_grad_batch_stride * batch_offset[:, None] +
                                output_grad_feat_stride * spatial_offset[None, :])
        output_grad_mask = batch_mask[:, None] & spatial_mask[None, :]

    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask)
    input_grad = -output_grad

    target_pointer += (target_batch_stride * batch_offset[:, None] +
                       target_spatial_stride * spatial_offset[None, :])
    target = tl.load(target_pointer,
                     mask=batch_mask[:, None] & spatial_mask[None, :])

    if weighted:
        weight = tl.load(weight_pointer + target,
                         mask=batch_mask[:, None] & spatial_mask[None, :])
        input_grad *= weight

        if reduction == 'mean':
            input_grad /= tl.load(sum_weights_pointer)

    elif reduction == 'mean':
        input_grad /= batch_dim * spatial_dim

    input_grad_pointer += (input_grad_feat_stride * target +
                           input_grad_batch_stride * batch_offset[:, None] +
                           input_grad_spatial_stride * spatial_offset[None, :])
    tl.store(input_grad_pointer, input_grad,
             mask=batch_mask[:, None] & spatial_mask[None, :])

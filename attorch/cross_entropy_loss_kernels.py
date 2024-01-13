"""
Kernels for cross entropy loss.
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
def cross_entropy_loss_forward_kernel(
    input_pointer, target_pointer, weight_pointer,
    sum_weights_pointer, output_pointer,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Measures the mean cross entropy loss between the input and target,
    with optional reweighing of each class.

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to a container the sum of the class weights is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        output_pointer: Pointer to a container the loss is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes BLOCK_SIZE_BATCH rows and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    target = tl.load(target_pointer + batch_offset, mask=batch_mask)

    pred_pointer = (input_pointer +
                    input_feat_stride * target +
                    input_batch_stride * batch_offset)
    input_pointer += (input_batch_stride * batch_offset[:, None] +
                      input_feat_stride * feat_offset[None, :])

    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :],
                    other=-float('inf'))
    pred = tl.load(pred_pointer, mask=batch_mask)
    mx = tl.max(input, axis=1)
    input -= mx[:, None]
    loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx

    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask)
        loss *= weight
        tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))

    else:
        loss /= batch_dim

    tl.store(output_pointer + batch_pid, tl.sum(loss))


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'feat_dim'],
)
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
                    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def cross_entropy_loss_backward_kernel(
    output_grad_pointer, target_pointer, input_pointer, weight_pointer,
    sum_weights_pointer, input_grad_pointer,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    input_grad_batch_stride, input_grad_feat_stride,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Calculates the input gradient of cross entropy loss.

    Args:
        output_grad_pointer: Pointer to the loss's output gradients.
            The output gradient must be a scalar.
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to the sum of the class weights if the classes were weighed.
            The sum of weights must be a scalar.
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        weighted: Flag for weighing each class.
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
    input_grad_pointer += (input_grad_batch_stride * batch_offset[:, None] +
                           input_grad_feat_stride * feat_offset[None, :])

    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :],
                    other=-float('inf'))
    input -= tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    softmax = numerator / tl.sum(numerator, axis=1)[:, None]

    output_grad = tl.load(output_grad_pointer)
    target = tl.load(target_pointer + batch_offset, mask=batch_mask)
    broadcasted_feat_offset = tl.broadcast_to(feat_offset[None, :],
                                              (BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT))
    broadcasted_target = tl.broadcast_to(target[:, None],
                                         (BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT))
    input_grad = output_grad * (softmax - (broadcasted_feat_offset == broadcasted_target))

    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask)
        sum_weights = tl.load(sum_weights_pointer)
        input_grad *= weight[:, None] / sum_weights

    else:
        input_grad /= batch_dim

    tl.store(input_grad_pointer, input_grad,
             mask=batch_mask[:, None] & feat_mask[None, :])

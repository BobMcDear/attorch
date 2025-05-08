"""
Kernels for p-norm-induced losses.
"""


import triton
import triton.language as tl

from .utils import element_wise_kernel_configs


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def p_loss_forward_kernel(
    input_pointer, target_pointer, output_pointer,
    param, size, p_loss: tl.constexpr, reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Measures the smooth L1, L1, or squared L2 norm of the difference between the input
    and target.

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [size].
        target_pointer: Pointer to the target.
            The target must be of shape [size].
        output_pointer: Pointer to a container the error is written to.
            The container must be of shape [size] if reduction is 'none',
            and otherwise of shape [size/BLOCK_SIZE].
        param: Parameter of loss function (i.e., beta for smooth L1).
        size: Number of elements in the input and target.
        p_loss: p-norm used to compute the error.
            Options are 0 for smooth L1, 1 for L1, and 2 for squared L2.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
            If a reduction method is specified, the reduced result of each
            program is written to a separate index in the output container,
            which should later be summed.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input = tl.load(input_pointer + offset, mask=mask).to(tl.float32)
    target = tl.load(target_pointer + offset, mask=mask).to(tl.float32)
    diff = input - target

    if p_loss == 0:
        error = tl.where(diff < param, 0.5 * diff * diff / param, tl.abs(diff) - 0.5 * param)

    elif p_loss == 1:
        error = tl.abs(diff)

    elif p_loss == 2:
        error = diff * diff

    if reduction == 'none':
        tl.store(output_pointer + offset, error, mask=mask)

    elif reduction == 'mean':
        tl.store(output_pointer + pid, tl.sum(error) / size)

    elif reduction == 'sum':
        tl.store(output_pointer + pid, tl.sum(error))


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def p_loss_backward_kernel(
    output_grad_pointer, input_pointer, target_pointer,
    input_grad_pointer, target_grad_pointer, param, size,
    p_loss: tl.constexpr, reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Calculates the input gradient of the smooth L1, L1, or L2 norm.

    Args:
        output_grad_pointer: Pointer to the error's output gradients.
            The output gradients must be a scalar or of shape [size].
        input_pointer: Pointer to the input.
            The input must be of shape [size].
        target_pointer: Pointer to the target.
            The target must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        target_grad_pointer: Pointer to a container the target's gradients are written to.
            The container must be of shape [size].
        param: Parameter of loss function (i.e., beta for smooth L1).
        size: Number of elements in the input and target.
        p_loss: p-norm used to compute the error whose gradient is calculated.
            Options are 0 for smooth L1, 1 for L1, and 2 for squared L2.
        reduction: Reduction strategy for the output whose gradient is calculated.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad_mask = None
    if reduction == 'none':
        output_grad_pointer += offset
        output_grad_mask = mask

    input = tl.load(input_pointer + offset, mask=mask).to(tl.float32)
    target = tl.load(target_pointer + offset, mask=mask).to(tl.float32)
    diff = input - target
    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask).to(tl.float32)

    if p_loss == 0:
        input_grad = tl.where(diff < param, diff / param, tl.where(0 <= diff, 1, -1))

    elif p_loss == 1:
        input_grad = tl.where(0 <= diff, 1, -1)

    elif p_loss == 2:
        input_grad = 2 * diff

    if reduction == 'mean':
        input_grad /= size

    input_grad *= output_grad
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)
    tl.store(target_grad_pointer + offset, -input_grad, mask=mask)

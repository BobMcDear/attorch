"""
Kernels for dropout.
"""


import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['size'],
)
@triton.jit
def dropout_forward_kernel(
    input_pointer, output_pointer, size,
    drop_p, seed,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Randomly zeroes elements in the input.

    Args:
        input_pointer: Pointer to the input to perform dropout on.
            The input must be of shape [size].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    random = tl.rand(seed, offset)
    input = tl.load(input_pointer + offset, mask=mask)
    output = tl.where(random < drop_p, 0, input / (1 - drop_p))
    tl.store(output_pointer + offset, output, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['size'],
)
@triton.jit
def dropout_backward_kernel(
    output_grad_pointer, input_grad_pointer, size,
    drop_p, seed,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Calculates the input gradient of dropout.

    Args:
        output_grad_pointer: Pointer to dropout's output gradients.
            The output container must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element used in dropout.
        seed: Seed for generating the dropout mask.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    random = tl.rand(seed, offset)
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input_grad = tl.where(random < drop_p, 0, output_grad / (1 - drop_p))
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)

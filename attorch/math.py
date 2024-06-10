"""
Pure math operations to be performed on loaded Triton tensors.
"""


import triton
import triton.language as tl

from .act_kernels import apply_act_func


@triton.jit
def accum_linear(accum, input1, input2,
                 fp16: tl.constexpr, tf32: tl.constexpr):
    """
    Accumulates matrix multiplications of input tensors for linear functions.

    Args:
        accum: Accumulator holding aggregation of matrix multiplications.
            The accumulator must be of shape [BLOCK_SIZE1, BLOCK_SIZE3].
        input1: First operand of matrix multiplication.
            The operand must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        input2: Second operand of matrix multiplication.
            The operand must be of shape [BLOCK_SIZE2, BLOCK_SIZE3].
        fp16: Flag for converting operands to FP16.
        tf32: Flag for performing matrix multiplication in TF32.

    Returns:
        Accumulator with the result of the new matrix multiplication added to it.
    """
    if fp16:
        input1 = input1.to(tl.float16)
        input2 = input2.to(tl.float16)

    return accum + tl.dot(input1, input2, allow_tf32=tf32)


@triton.jit
def glu(input1, input2, act_func: tl.constexpr):
    """
    Applies the gated linear unit with an arbitrary activation function
    to the input.

    Args:
        input1: First half of input to gate.
            The first half must be of the same shape as the second half.
        input2: Second half of input to gate.
            The second half must be of the same shape as the first half.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', and 'silu'.

    Args:
        Input transformed by the gated linear unit
        with an arbitrary activation function.
    """
    return input1 * apply_act_func(input2, None, None, None, act_func, False)


@triton.jit
def softmax(input,
            log: tl.constexpr):
    """
    Normalizes the input using softmax along the last dimension.

    Args:
        input: Input to normalize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        log: Flag for indicating if the log of softmax should be taken.

    Returns:
        Input normalized by softmax.
    """
    input = input.to(tl.float32)

    input = input - tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=1)[:, None]

    if log:
        output = input - tl.log(denominator)

    else:
        output = numerator / denominator

    return output

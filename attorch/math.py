"""
Pure math operations to be performed on loaded Triton tensors.
"""


import triton
import triton.language as tl


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

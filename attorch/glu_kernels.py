"""
Kernels for gated linear units with arbitrary activation functions.
"""


import triton
import triton.language as tl

from .act_kernels import apply_act_func, apply_act_func_grad
from .utils import element_wise_kernel_configs


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def glu_forward_kernel(
    input1_pointer, input2_pointer, output_pointer, size, param,
    act_func: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Applies the gated linear unit with an arbitrary activation function
    to the input.

    Args:
        input1_pointer: Pointer to the first half of the input to gate.
            The first half must be contiguous and contain size elements.
        input2_pointer: Pointer to the second half of the input to gate.
            The second half must be contiguous and contain size elements.
        output_pointer: Pointer to a container the result is written to.
            The container must be contiguous and contain size elements.
        size: Number of elements in each half of the input.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input1 = tl.load(input1_pointer + offset, mask=mask)
    input2 = tl.load(input2_pointer + offset, mask=mask)

    output = input1 * apply_act_func(input2, None, None, None, param,
                                     act_func, False)
    tl.store(output_pointer + offset, output, mask=mask)


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def glu_backward_kernel(
    output_grad_pointer, input1_pointer, input2_pointer,
    input1_grad_pointer, input2_grad_pointer, size, param,
    act_func: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Calculates the input gradient of the gated linear unit.

    Args:
        output_grad_pointer: Pointer to the unit's output gradients.
            The output gradients must be contiguous and contain size elements.
        input1_pointer: Pointer to the first half of the input that was gated.
            The first half must be contiguous and contain size elements.
        input2_pointer: Pointer to the second half of the input that was gated.
            The second half must be contiguous and contain size elements.
        input1_grad_pointer: Pointer to a container the first half's gradients are written to.
            The container must be contiguous and contain size elements.
        input2_grad_pointer: Pointer to a container the second half's gradients are written to.
            The container must be contiguous and contain size elements.
        size: Number of elements in each half of the input.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input1 = tl.load(input1_pointer + offset, mask=mask)
    input2 = tl.load(input2_pointer + offset, mask=mask)

    input1_grad = output_grad * apply_act_func(input2, None, None, None, param,
                                               act_func, False)
    input2_grad = output_grad * input1 * apply_act_func_grad(1, input2,
                                                             None, None, None,
                                                             param, act_func,
                                                             False)

    tl.store(input1_grad_pointer + offset, input1_grad, mask=mask)
    tl.store(input2_grad_pointer + offset, input2_grad, mask=mask)

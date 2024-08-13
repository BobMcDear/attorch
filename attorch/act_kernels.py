"""
Kernels for activation functions with fused dropout.
"""


import triton
import triton.language as tl

from .dropout_kernels import apply_dropout, apply_dropout_grad
from .utils import element_wise_kernel_configs


@triton.jit
def sigmoid(input):
    """
    Applies sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by sigmoid.
    """
    return (1 / (1 + tl.exp(-input)))


@triton.jit
def sigmoid_grad(input):
    """
    Calculates the gradient of sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of sigmoid.
    """
    output_sigmoid = sigmoid(input)
    return output_sigmoid * (1 - output_sigmoid)


@triton.jit
def tanh(input):
    """
    Applies tanh to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by tanh.
    """
    return 2 * sigmoid(2 * input) - 1


@triton.jit
def tanh_grad(input):
    """
    Calculates the gradient of tanh.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of tanh.
    """
    output_tanh = tanh(input)
    return 1 - output_tanh * output_tanh


@triton.jit
def relu(input):
    """
    Applies ReLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by ReLU.
    """
    return tl.maximum(0, input)


@triton.jit
def relu_grad(input):
    """
    Calculates the gradient of ReLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of ReLU.
    """
    return tl.where(input <= 0, 0, 1)


@triton.jit
def gelu(input):
    """
    Applies GELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input


@triton.jit
def gelu_grad(input):
    """
    Calculates the gradient of GELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    cdf_grad = 0.39894228 * tl.exp(-0.5 * input * input)
    return (cdf_grad * input + cdf)


@triton.jit
def silu(input):
    """
    Applies SiLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by SiLU.
    """
    return (input * sigmoid(input))


@triton.jit
def silu_grad(input):
    """
    Calculates the gradient of SiLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of SiLU.
    """
    output_sigmoid = sigmoid(input)
    return (output_sigmoid * (input * (1 - output_sigmoid) + 1))


@triton.jit
def apply_act_func(input, drop_p, seed, offset,
                   act_func: tl.constexpr, dropout: tl.constexpr):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', and 'silu'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Input transformed by the desired activation function,
        potentially with fused dropout.
    """
    # All activation functions, except ReLU, require FP32 inputs.
    if act_func != 'relu':
        input = input.to(tl.float32)

    if act_func == 'sigmoid':
        output = sigmoid(input)

    elif act_func == 'tanh':
        output = tanh(input)

    elif act_func == 'relu':
        output = relu(input)

    elif act_func == 'gelu':
        output = gelu(input)

    elif act_func == 'silu':
        output = silu(input)

    if dropout:
        output = apply_dropout(output, drop_p, seed, offset)

    return output


@triton.jit
def apply_act_func_grad(output_grad, input, drop_p, seed, offset,
                        act_func: tl.constexpr, dropout: tl.constexpr):
    """
    Calculates the gradient of an activation function.

    Args:
        output_grad: Output gradients. The output gradients must be
            loaded and cannot be a pointer.
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', and 'silu'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Gradient of the desired activation function.
    """
    # All activation function derivatives, except ReLU's, require FP32 inputs.
    if act_func != 'relu':
        input = input.to(tl.float32)

    if act_func == 'sigmoid':
        output = sigmoid_grad(input)

    elif act_func == 'tanh':
        output = tanh_grad(input)

    elif act_func == 'relu':
        output = relu_grad(input)

    elif act_func == 'gelu':
        output = gelu_grad(input)

    elif act_func == 'silu':
        output = silu_grad(input)

    if dropout:
        output_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)

    return output_grad * output


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def act_func_forward_kernel(
    input_pointer, output_pointer, size,
    drop_p, seed,
    act_func: tl.constexpr, dropout: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input_pointer: Pointer to the input to transform.
            The input must be of shape [size].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', and 'silu'.
        dropout: Flag for performing dropout on the activation output.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input = tl.load(input_pointer + offset, mask=mask)
    tl.store(output_pointer + offset,
             apply_act_func(input, drop_p, seed, offset, act_func, dropout),
             mask=mask)


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def act_func_backward_kernel(
    output_grad_pointer, input_pointer, input_grad_pointer, size,
    drop_p, seed,
    act_func: tl.constexpr, dropout: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Calculates the input gradient of an activation function.

    Args:
        output_grad_pointer: Pointer to the activation's output gradients.
            The output gradients must be of shape [size].
        input_pointer: Pointer to the activation's input.
            The input must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', and 'silu'.
        dropout: Flag for performing dropout on the activation output.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input = tl.load(input_pointer + offset, mask=mask)

    tl.store(input_grad_pointer + offset,
             apply_act_func_grad(output_grad, input, drop_p, seed, offset, act_func, dropout),
             mask=mask)

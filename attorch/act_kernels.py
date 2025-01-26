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
def logsigmoid(input):
    """
    Applies the log of sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by the log of sigmoid.
    """
    return tl.log(sigmoid(input))


@triton.jit
def logsigmoid_grad(input):
    """
    Calculates the gradient of the log of sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of the log of sigmoid.
    """
    return (1 / (1 + tl.exp(input)))


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
def relu6(input):
    """
    Applies ReLU6 to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by ReLU6.
    """
    return tl.minimum(relu(input), 6)


@triton.jit
def relu6_grad(input):
    """
    Calculates the gradient of ReLU6.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of ReLU6.
    """
    return tl.where((0 < input) & (input < 6), 1, 0)


@triton.jit
def hardsigmoid(input):
    """
    Applies hard sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard sigmoid.
    """
    return tl.maximum(0, tl.minimum(1, input / 6 + 0.5))


@triton.jit
def hardsigmoid_grad(input):
    """
    Calculates the gradient of hard sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard sigmoid.
    """
    return tl.where((-3 < input) & (input < 3), 1 / 6, 0)


@triton.jit
def hardtanh(input):
    """
    Applies hard tanh to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard tanh.
    """
    return tl.maximum(-1, tl.minimum(1, input))


@triton.jit
def hardtanh_grad(input):
    """
    Calculates the gradient of hard tanh.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard tanh.
    """
    return tl.where((-1 < input) & (input < 1), 1, 0)


@triton.jit
def hardswish(input):
    """
    Applies hard Swish to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard Swish.
    """
    return input * relu6(input + 3) / 6


@triton.jit
def hardswish_grad(input):
    """
    Calculates the gradient of hard Swish.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard Swish.
    """
    return (relu6(input + 3) + input * relu6_grad(input + 3)) / 6


@triton.jit
def selu(input):
    """
    Applies SELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by SELU.
    """
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * (tl.maximum(0, input) +
                    tl.minimum(0, alpha * (tl.exp(input) - 1)))


@triton.jit
def selu_grad(input):
    """
    Calculates the gradient of SELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of SELU.
    """
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * tl.where(input <= 0, alpha * tl.exp(input), 1)


@triton.jit
def mish(input):
    """
    Applies Mish to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by Mish.
    """
    return input * tanh(tl.log(1 + tl.exp(input)))


@triton.jit
def mish_grad(input):
    """
    Calculates the gradient of Mish.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of Mish.
    """
    exp = tl.exp(input)
    delta = exp * (exp + 2) + 2
    return (exp * (exp * ((4 * input + 6) + exp * (exp + 4)) + 4 * (input + 1)) /
            (delta * delta))


@triton.jit
def leaky_relu(input, negative_slope):
    """
    Applies leaky ReLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        negative_slope: Slope of the negative component.

    Returns:
        Input transformed by leaky ReLU.
    """
    return relu(input) + negative_slope * tl.minimum(0, input)


@triton.jit
def leaky_relu_grad(input, negative_slope):
    """
    Calculates the gradient of leaky ReLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        negative_slope: Slope of the negative component.

    Returns:
        Gradient of leaky ReLU.
    """
    return tl.where(input <= 0, negative_slope, 1)


@triton.jit
def apply_act_func(input, drop_p, seed, offset, param,
                   act_func: tl.constexpr, dropout: tl.constexpr):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Input transformed by the desired activation function,
        potentially with fused dropout.
    """
    if act_func == 'sigmoid':
        input = input.to(tl.float32)
        output = sigmoid(input)

    if act_func == 'logsigmoid':
        input = input.to(tl.float32)
        output = logsigmoid(input)

    elif act_func == 'tanh':
        input = input.to(tl.float32)
        output = tanh(input)

    elif act_func == 'relu':
        output = relu(input)

    elif act_func == 'gelu':
        input = input.to(tl.float32)
        output = gelu(input)

    elif act_func == 'silu':
        input = input.to(tl.float32)
        output = silu(input)

    elif act_func == 'relu6':
        output = relu6(input)

    elif act_func == 'hardsigmoid':
        output = hardsigmoid(input)

    elif act_func == 'hardtanh':
        output = hardtanh(input)

    elif act_func == 'hardswish':
        output = hardswish(input)

    elif act_func == 'selu':
        input = input.to(tl.float32)
        output = selu(input)

    elif act_func == 'mish':
        input = input.to(tl.float32)
        output = mish(input)

    elif act_func == 'leaky_relu':
        output = leaky_relu(input, param)

    if dropout:
        output = apply_dropout(output, drop_p, seed, offset)

    return output


@triton.jit
def apply_act_func_grad(output_grad, input, drop_p, seed, offset, param,
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
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Gradient of the desired activation function.
    """
    if act_func == 'sigmoid':
        input = input.to(tl.float32)
        output = sigmoid_grad(input)

    if act_func == 'logsigmoid':
        input = input.to(tl.float32)
        output = logsigmoid_grad(input)

    elif act_func == 'tanh':
        input = input.to(tl.float32)
        output = tanh_grad(input)

    elif act_func == 'relu':
        output = relu_grad(input)

    elif act_func == 'gelu':
        input = input.to(tl.float32)
        output = gelu_grad(input)

    elif act_func == 'silu':
        input = input.to(tl.float32)
        output = silu_grad(input)

    elif act_func == 'relu6':
        output = relu6_grad(input)

    elif act_func == 'hardsigmoid':
        output = hardsigmoid_grad(input)

    elif act_func == 'hardtanh':
        output = hardtanh_grad(input)

    elif act_func == 'hardswish':
        output = hardswish_grad(input)

    elif act_func == 'selu':
        input = input.to(tl.float32)
        output = selu_grad(input)

    elif act_func == 'mish':
        input = input.to(tl.float32)
        output = mish_grad(input)

    elif act_func == 'leaky_relu':
        output = leaky_relu_grad(input, param)

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
    drop_p, seed, param,
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
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input = tl.load(input_pointer + offset, mask=mask)
    tl.store(output_pointer + offset,
             apply_act_func(input, drop_p, seed, offset,
                            param, act_func, dropout),
             mask=mask)


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def act_func_backward_kernel(
    output_grad_pointer, input_pointer, input_grad_pointer, size,
    drop_p, seed, param,
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
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
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
             apply_act_func_grad(output_grad, input, drop_p, seed,
                                 offset, param, act_func, dropout),
             mask=mask)

"""
Kernels for activation functions.
"""


import triton
import triton.language as tl


@triton.jit
def sigmoid(input):
    """
    Applies sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by sigmoid.
    """
    return 1 / (1 + tl.exp(-input))


@triton.jit
def sigmoid_grad(input):
    """
    Calculates the gradient of sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of sigmoid.
    """
    return sigmoid(input) * (1 - sigmoid(input))


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
    return cdf_grad * input + cdf


@triton.jit
def apply_act_func(input, act_func: tl.constexpr):
    """
    Applies an activation function to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', and 'gelu'.

    Returns:
        Input transformed by the desired activation function.
    """
    if act_func == 'sigmoid':
        output = sigmoid(input)

    elif act_func == 'tanh':
        output = tanh(input)

    elif act_func == 'relu':
        output = relu(input)

    elif act_func == 'gelu':
        output = gelu(input)

    return output


@triton.jit
def apply_act_func_grad(input, act_func: tl.constexpr):
    """
    Calculates the gradient of an activation function.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'tanh', 'relu', and 'gelu'.

    Returns:
        Gradient of the desired activation function.
    """
    if act_func == 'sigmoid':
        output = sigmoid_grad(input)

    elif act_func == 'tanh':
        output = tanh_grad(input)

    elif act_func == 'relu':
        output = relu_grad(input)

    elif act_func == 'gelu':
        output = gelu_grad(input)

    return output


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
def act_func_forward_kernel(
    input_pointer, output_pointer,
    feat_dim,
    input_batch_stride, input_feat_stride,
    act_func: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Applies an activation function to the input.

    Args:
        input_pointer: Pointer to the input to transform.
            The input must be of shape [batch_dim, feat_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim] and contiguous.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', and 'gelu'.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes a single row and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)
    feat_pid = tl.program_id(axis=1)

    feat_offset = feat_pid * BLOCK_SIZE_FEAT + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < feat_dim

    input_pointer += (batch_pid * input_batch_stride +
                      feat_offset * input_feat_stride)
    output_pointer += batch_pid * feat_dim + feat_offset

    input = tl.load(input_pointer, mask=feat_mask)
    tl.store(output_pointer, apply_act_func(input, act_func), mask=feat_mask)


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
def act_func_backward_kernel(
    output_grad_pointer, input_pointer, input_grad_pointer,
    feat_dim,
    output_grad_batch_stride, output_grad_feat_stride,
    input_batch_stride, input_feat_stride,
    act_func: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
    ):
    """
    Calculates the input gradient of an activation function.

    Args:
        output_grad_pointer: Pointer to the activation's output gradients.
            The output container must be of shape [batch_dim, feat_dim].
        input_pointer: Pointer to the activation's input.
            The input must be of shape [batch_dim, feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim] and contiguous.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'tanh', 'relu', and 'gelu'.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    # This program processes a single row and BLOCK_SIZE_FEAT columns.
    batch_pid = tl.program_id(axis=0)
    feat_pid = tl.program_id(axis=1)

    feat_offset = feat_pid * BLOCK_SIZE_FEAT + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < feat_dim

    output_grad_pointer += (batch_pid * output_grad_batch_stride +
                            feat_offset * output_grad_feat_stride)
    input_pointer += (batch_pid * input_batch_stride +
                      feat_offset * input_feat_stride)
    input_grad_pointer += batch_pid * feat_dim + feat_offset

    output_grad = tl.load(output_grad_pointer, mask=feat_mask)
    input = tl.load(input_pointer, mask=feat_mask)

    input_grad = output_grad * apply_act_func_grad(input, act_func)
    tl.store(input_grad_pointer, input_grad, mask=feat_mask)

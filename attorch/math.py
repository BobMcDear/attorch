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
def glu(input1, input2, param, act_func: tl.constexpr):
    """
    Applies the gated linear unit with an arbitrary activation function
    to the input.

    Args:
        input1: First half of input to gate.
            The first half must be of the same shape as the second half.
        input2: Second half of input to gate.
            The second half must be of the same shape as the first half.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', and 'celu'.
        param: Parameter in the case of parameterized activation functions.

    Args:
        Input transformed by the gated linear unit
        with an arbitrary activation function.
    """
    return input1 * apply_act_func(input2, None, None, None, param, act_func, False)


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


@triton.jit
def calc_mean_and_inv_std(input, last_dim, eps,
                          last_dim_mask: tl.constexpr):
    """
    Calculates the mean and inverse standard deviation of the input
    along the last dimension.

    Args:
        input: Input whose mean and inverse standard deviation are calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        last_dim: Size of the last dimension of input.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Mean and inverse standard deviation of the input.
    """
    input = input.to(tl.float32)

    mean = tl.sum(input, axis=1) / last_dim
    diff = tl.where(last_dim_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / last_dim + eps)

    return mean, inv_std


@triton.jit
def update_welford(input, prev_count, prev_mean, prev_var, curr_count,
                   mask: tl.constexpr):
    """
    Updates count, mean, and variance (M2) statistics for Welford's algorithm.

    Args:
        input: Input used to update statistics.
            The input must be of the same shape as the mask.
        prev_count: Previous count statistic to update.
        prev_mean: Previous mean statistic to update.
        prev_var: Previous variance (M2) statistic to update.
        curr_count: Count of elements in current input.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Updated count, mean, and variance (M2) statistics
    """
    input = input.to(tl.float32)

    count = prev_count + curr_count
    mean = (tl.sum(input) - curr_count * prev_mean) / count
    deltas = tl.where(mask, (input - mean) * (input - prev_mean), 0.)
    var = prev_var + tl.sum(deltas)

    return count, mean, var


@triton.jit
def update_ema(prev_ema, new_val, momentum):
    """
    Updates exponential moving average.

    Args:
        prev_ema: Previous exponential moving average.
        new_val: Value used to update the exponential moving average.
        momentum: Momentum.

    Returns:
        Updated running statistic.
    """
    return (1 - momentum) * prev_ema + momentum * new_val


@triton.jit
def standardize(input, mean, inv_std, weight, bias):
    """
    Standardizes the input given its mean and inverse standard deviation,
    multiplies the result by weights, and adds a bias vector.

    Args:
        input: Input to standardize.
        mean: Mean of input.
        inv_std: Inverse standard deviation of input.
        weight: Weight multiplied by the standardized input.
        bias: Bias added to the result of the weight multiplication.

    Returns:
        Standardized input.
    """
    return weight * inv_std * (input - mean) + bias


@triton.jit
def calc_p_loss(input, target, size,
                p_loss: tl.constexpr, reduction: tl.constexpr):
    """
    Measures the L1 or squared L2 norm of the difference between the input
    and target (i.e., mean absolute error or mean squared error).

    Args:
        input: Input.
            The input must be of shape [BLOCK_SIZE].
        target: Target.
            The target must be of shape [BLOCK_SIZE].
        size: Number of elements in the input and target.
            This value is used only if reduction is 'mean'.
        p_loss: p-norm used to compute the error.
            Options are 1 for MAE and 2 for MSE.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.

    Returns:
        Error.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    diff = input - target

    if p_loss == 1:
        error = tl.abs(diff)

    elif p_loss == 2:
        error = diff * diff

    if reduction == 'none':
        output = error

    elif reduction == 'mean':
        output = tl.sum(error) / size

    elif reduction == 'sum':
        output = tl.sum(error)

    return output


@triton.jit
def nll_loss(input, size,
             reduction: tl.constexpr):
    """
    Measures the negative log likelihood loss given log-probabilities of target class.

    Args:
        input: Input containing predicted log-probabilities corresponding to target class.
            The input can have arbitrary shape.
        size: Number of elements in the input.
            This value is used only if reduction is 'mean'.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.

    Returns:
        Loss.
    """
    input = input.to(tl.float32)

    if reduction == 'none':
        output = -input

    elif reduction == 'mean':
        output = -tl.sum(input) / size

    elif reduction == 'sum':
        output = -tl.sum(input)

    return output


@triton.jit
def cross_entropy_loss(input, pred):
    """
    Measures the per-row cross entropy loss given
    input and predicted logits corresponding to target class.

    Args:
        input: Input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        pred: Predicted logits corresponding to target class.
            The predictions must be of shape [BLOCK_SIZE1].

    Returns:
        Loss.
    """
    input = input.to(tl.float32)
    pred = pred.to(tl.float32)

    mx = tl.max(input, axis=1)
    input -= mx[:, None]
    loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx

    return loss

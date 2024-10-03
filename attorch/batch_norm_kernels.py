"""
Kernels for batch normalization with residual addition and fused activation.
"""


from typing import Dict

import triton
import triton.language as tl
from triton import next_power_of_2

from .act_kernels import apply_act_func
from .utils import warps_kernel_configs


def BLOCK_SIZE_SPATIAL_heuristic(args: Dict) -> int:
    """
    Approximates an appropriate spatial block size for batch normalization
    using a heuristic.

    Args:
        args: Arguments to batch normalization kernel.

    Returns:
        Appropriate spatial block size.
    """
    # Preferrably, the batch and spatial dimensions would both be loaded,
    # and normalization would be conducted in one step.
    # However, for large inputs, that is not feasible given memory constraints.
    # Thus, a maximum of 16384 elements are loaded at once.
    BLOCK_SIZE_BATCH = next_power_of_2(args['batch_dim'])
    BLOCK_SIZE_SPATIAL = next_power_of_2(args['spatial_dim'])
    return min(BLOCK_SIZE_SPATIAL, max(1, 2 ** 14 // BLOCK_SIZE_BATCH))


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'spatial_dim'],
    restore_value=['running_mean_pointer', 'running_var_pointer']
)
@triton.heuristics({'BLOCK_SIZE_BATCH': lambda args: next_power_of_2(args['batch_dim']),
                    'BLOCK_SIZE_SPATIAL': BLOCK_SIZE_SPATIAL_heuristic})
@triton.jit
def batch_norm_forward_kernel(
    input_pointer, weight_pointer, bias_pointer,
    mean_pointer, inv_std_pointer,
    pre_act_add_pointer, pre_act_pointer, output_pointer,
    running_mean_pointer, running_var_pointer,
    batch_dim, spatial_dim,
    input_batch_stride, input_feat_stride, input_spatial_stride,
    pre_act_add_batch_stride, pre_act_add_feat_stride, pre_act_add_spatial_stride,
    pre_act_batch_stride, pre_act_feat_stride, pre_act_spatial_stride,
    output_batch_stride, output_feat_stride, output_spatial_stride,
    momentum, eps, param,
    affine: tl.constexpr, save_stats: tl.constexpr,
    track_running_stats: tl.constexpr, is_train: tl.constexpr,
    add_pre_act: tl.constexpr, act_func: tl.constexpr, save_pre_act: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_SPATIAL: tl.constexpr,
    ):
    """
    Batch-normalizes the input, optionally adding a residual and fusing an activation function.

    Args:
        input_pointer: Pointer to the input to layer-normalize.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        weight_pointer: Pointer to optional weights for affine transform.
            The weights, if provided, must be of shape [feat_dim].
        bias_pointer: Pointer to an optional bias vector for affine transform.
            The bias vector, if provided, must be of shape [feat_dim].
        mean_pointer: Pointer to an optional container the input's mean
            is written to if save_stats is True.
            The container, if provided, must be of shape [feat_dim].
        inv_std_pointer: Pointer to an optional container the input's inverse
            standard deviation is written to if save_stats is True.
            The container, if provided, must be of shape [feat_dim].
        pre_act_add_pointer: Pointer to an optional residual added to the pre-activation result.
            The residual, if provided, must be of shape [batch_dim, feat_dim, spatial_dim].
        pre_act_pointer: Pointer to an optional container the pre-activation input
            is written to if act_func is not None and save_pre_act is True.
            The container, if provided, must be of shape [batch_dim, feat_dim, spatial_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim].
        running_mean_pointer: Pointer to an optional container the input's running
            mean is written to if track_running_stats and is_train are True.
            The container, if provided, must be of shape [feat_dim].
        running_var_pointer: Pointer to an optional container the input's running
            variance is written to if track_running_stats and is_train are True.
            The container, if provided, must be of shape [feat_dim].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        pre_act_add_batch_stride: Stride necessary to jump one element along the
            residual's batch dimension.
        pre_act_add_out_feat_stride: Stride necessary to jump one element along the
            residual's feature dimension.
        pre_act_add_spatial_stride: Stride necessary to jump one element along the
            residual's spatial dimension.
        pre_act_batch_stride: Stride necessary to jump one element along the
            pre-activation input container's batch dimension.
        pre_act_out_feat_stride: Stride necessary to jump one element along the
            pre-activation input container's feature dimension.
        pre_act_spatial_stride: Stride necessary to jump one element along the
            pre-activation input container's spatial dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        output_spatial_stride: Stride necessary to jump one element along the
            output container's spatial dimension.
        momentum: Momentum for the running mean and variance.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        param: Parameter in the case of parameterized activation functions.
        affine: Flag for performing an affine transformation on the normalized output.
        save_stats: Flag for saving the mean and standard deviation.
        track_running_stats: Flag for tracking running mean and variance if
            is_train is also True.
        is_train: Flag indicating if the model is in training mode.
        add_pre_act: Flag for adding the residual to the pre-activation result.
        act_func: Name of activation function to apply, with None for identity.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        save_pre_act: Flag for saving the pre-activation input.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    feat_pid = tl.program_id(axis=0)

    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim

    if is_train or not track_running_stats:
        count = 0
        mean = 0.0
        var = 0.0

        for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
            spatial_offset = (block_ind * BLOCK_SIZE_SPATIAL +
                              tl.arange(0, BLOCK_SIZE_SPATIAL))
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (input_pointer +
                                  input_feat_stride * feat_pid +
                                  input_batch_stride * batch_offset[:, None] +
                                  input_spatial_stride * spatial_offset[None, :])
            curr_input = tl.load(curr_input_pointer,
                                 mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)

            spatial_count = min(BLOCK_SIZE_SPATIAL, spatial_dim - block_ind * BLOCK_SIZE_SPATIAL)
            curr_count = spatial_count * batch_dim
            count += curr_count

            prev_mean = mean
            mean += (tl.sum(curr_input) - curr_count * mean) / count
            deltas = tl.where(batch_mask[:, None] & spatial_mask[None, :],
                              (curr_input - mean) * (curr_input - prev_mean), 0.)
            var += tl.sum(deltas)

        var /= count
        inv_std = tl.rsqrt(var + eps)

        if save_stats:
            tl.store(feat_pid + mean_pointer, mean)
            tl.store(feat_pid + inv_std_pointer, inv_std)

        if track_running_stats:
            running_mean_pointer += feat_pid
            running_var_pointer += feat_pid

            running_mean = tl.load(running_mean_pointer)
            running_var = tl.load(running_var_pointer)

            n = batch_dim * spatial_dim
            tl.store(running_mean_pointer,
                     (1 - momentum) * running_mean + momentum * mean)
            tl.store(running_var_pointer,
                     (1 - momentum) * running_var + momentum * var * n / (n - 1))

    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = tl.rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        bias = tl.load(feat_pid + bias_pointer)

    else:
        weight = 1.
        bias = 0.

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = (block_ind * BLOCK_SIZE_SPATIAL +
                          tl.arange(0, BLOCK_SIZE_SPATIAL))
        spatial_mask = spatial_offset < spatial_dim

        curr_input_pointer = (input_pointer +
                              input_feat_stride * feat_pid +
                              input_batch_stride * batch_offset[:, None] +
                              input_spatial_stride * spatial_offset[None, :])
        curr_output_pointer = (output_pointer +
                               output_feat_stride * feat_pid +
                               output_batch_stride * batch_offset[:, None] +
                               output_spatial_stride * spatial_offset[None, :])

        curr_input = tl.load(curr_input_pointer,
                             mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
        output = weight * (curr_input - mean) * inv_std + bias

        if add_pre_act:
            curr_pre_act_add_pointer = (pre_act_add_pointer +
                                        pre_act_add_feat_stride * feat_pid +
                                        pre_act_add_batch_stride * batch_offset[:, None] +
                                        pre_act_add_spatial_stride * spatial_offset[None, :])
            curr_pre_act_add = tl.load(curr_pre_act_add_pointer,
                                       mask=batch_mask[:, None] & spatial_mask[None, :])
            output += curr_pre_act_add

        if act_func is not None:
            if save_pre_act:
                curr_pre_act_pointer = (pre_act_pointer +
                                        pre_act_feat_stride * feat_pid +
                                        pre_act_batch_stride * batch_offset[:, None] +
                                        pre_act_spatial_stride * spatial_offset[None, :])
                tl.store(curr_pre_act_pointer, output,
                         mask=batch_mask[:, None] & spatial_mask[None, :])

            output = apply_act_func(output, None, None, None, param, act_func, False)

        tl.store(curr_output_pointer, output,
                 mask=batch_mask[:, None] & spatial_mask[None, :])


@triton.autotune(
    configs=warps_kernel_configs(),
    key=['batch_dim', 'spatial_dim'],
)
@triton.heuristics({'BLOCK_SIZE_BATCH': lambda args: next_power_of_2(args['batch_dim']),
                    'BLOCK_SIZE_SPATIAL': BLOCK_SIZE_SPATIAL_heuristic})
@triton.jit
def batch_norm_backward_kernel(
    output_grad_pointer, input_pointer, mean_pointer, inv_std_pointer, weight_pointer,
    input_grad_pointer, weight_grad_pointer, bias_grad_pointer,
    batch_dim, spatial_dim,
    output_grad_batch_stride, output_grad_feat_stride, output_grad_spatial_stride,
    input_batch_stride, input_feat_stride, input_spatial_stride,
    input_grad_batch_stride, input_grad_feat_stride, input_grad_spatial_stride,
    affine: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_SPATIAL: tl.constexpr,
    ):
    """
    Calculates the input gradient of batch normalization.

    Args:
        output_grad_pointer: Pointer to layer normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim, spatial_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        mean_pointer: Pointer to the input's mean.
            The mean should be of shape [feat_dim].
        inv_std_pointer: Pointer to the input's inverse standard deviation.
            The inverse standard deviation should be of shape [feat_dim].
        weight_pointer: Pointer to optional weights if affine transform occurred.
            The weights, if provided, must be of shape [feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim].
        weight_grad_pointer: Pointer to an optional container the weights' gradients
            are written to if scale_by_weight is True.
            The container, if provided, must be of shape [feat_dim].
        bias_grad_pointer: Pointer to an optional container the bias vector's gradients
            are written to if scale_by_weight is True.
            The container, if provided, must be of shape [feat_dim].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        output_grad_spatial_stride: Stride necessary to jump one element along the
            output gradients' spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        input_grad_spatial_stride: Stride necessary to jump one element along the
            input gradient container's spatial dimension.
        affine: Flag for performing an affine transformation on the normalized output.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    feat_pid = tl.program_id(axis=0)

    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim

    mean = tl.load(feat_pid + mean_pointer)
    inv_std = tl.load(feat_pid + inv_std_pointer)

    term1 = 0.0
    term2 = 0.0

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = (block_ind * BLOCK_SIZE_SPATIAL +
                          tl.arange(0, BLOCK_SIZE_SPATIAL))
        spatial_mask = spatial_offset < spatial_dim

        curr_output_grad_pointer = (output_grad_pointer +
                                    output_grad_feat_stride * feat_pid +
                                    output_grad_batch_stride * batch_offset[:, None] +
                                    output_grad_spatial_stride * spatial_offset[None, :])
        curr_input_pointer = (input_pointer +
                              input_feat_stride * feat_pid +
                              input_batch_stride * batch_offset[:, None] +
                              input_spatial_stride * spatial_offset[None, :])

        curr_input = tl.load(curr_input_pointer,
                             mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(curr_output_grad_pointer,
                                   mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)

        term1 += tl.sum(curr_pre_lin * curr_output_grad)
        term2 += tl.sum(curr_output_grad)

    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        weight_grad = 0.0
        bias_grad = 0.0

    else:
        weight = 1.

    count = batch_dim * spatial_dim
    term1 *= weight / count
    term2 *= weight / count

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = (block_ind * BLOCK_SIZE_SPATIAL +
                          tl.arange(0, BLOCK_SIZE_SPATIAL))
        spatial_mask = spatial_offset < spatial_dim

        curr_output_grad_pointer = (output_grad_pointer +
                                    output_grad_feat_stride * feat_pid +
                                    output_grad_batch_stride * batch_offset[:, None] +
                                    output_grad_spatial_stride * spatial_offset[None, :])
        curr_input_pointer = (input_pointer +
                              input_feat_stride * feat_pid +
                              input_batch_stride * batch_offset[:, None] +
                              input_spatial_stride * spatial_offset[None, :])
        curr_input_grad_pointer = (input_grad_pointer +
                                   input_grad_feat_stride * feat_pid +
                                   input_grad_batch_stride * batch_offset[:, None] +
                                   input_grad_spatial_stride * spatial_offset[None, :])

        curr_input = tl.load(curr_input_pointer,
                             mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(curr_output_grad_pointer,
                                   mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
        curr_input_grad = inv_std * (weight * curr_output_grad - (term1 * curr_pre_lin + term2))
        tl.store(curr_input_grad_pointer, curr_input_grad,
                 mask=batch_mask[:, None] & spatial_mask[None, :])

        if affine:
            weight_grad += tl.sum(curr_pre_lin * curr_output_grad)
            bias_grad += tl.sum(curr_output_grad)

    if affine:
        tl.store(feat_pid + weight_grad_pointer, weight_grad)
        tl.store(feat_pid + bias_grad_pointer, bias_grad)

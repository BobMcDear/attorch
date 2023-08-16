"""
Kernels for linear layer with fused activation.
"""


import triton
import triton.language as tl

from .act_kernels import apply_act_func


def linear_forward_config(
    BLOCK_SIZE_BATCH: int,
    BLOCK_SIZE_IN_FEAT: int,
    BLOCK_SIZE_OUT_FEAT: int,
    GROUP_SIZE_BATCH: int = 8,
    n_warps: int = 4,
    n_stages: int = 2,
    ) -> triton.Config:
    """
    Creates a triton.Config object for linear_kernel
    given meta-parameters for auto-tuning.

    Args:
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
        GROUP_SIZE_BATCH: Group size across the batch dimension.
        n_warps: Number of warps to use for the kernel when compiled for GPUs.
        n_stages: Number of stages the compiler uses to software-pipeline.

    Returns:
        Kernel configuration.
    """
    return triton.Config({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH,
                          'BLOCK_SIZE_IN_FEAT': BLOCK_SIZE_IN_FEAT,
                          'BLOCK_SIZE_OUT_FEAT': BLOCK_SIZE_OUT_FEAT,
                          'GROUP_SIZE_BATCH': GROUP_SIZE_BATCH},
                          num_warps=n_warps, num_stages=n_stages)


@triton.autotune(
    configs=[
        linear_forward_config(32, 32, 32, n_warps=2, n_stages=2),
        linear_forward_config(64, 32, 32, n_warps=2, n_stages=5),
        linear_forward_config(64, 32, 128, n_warps=4, n_stages=4),
        linear_forward_config(64, 32, 256, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 32, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 64, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 128, n_warps=4, n_stages=4),
        linear_forward_config(128, 64, 256, n_warps=8, n_stages=3),
    ],
    key=['batch_dim', 'in_feat_dim', 'out_feat_dim'],
)
@triton.jit
def linear_forward_kernel(
    input_pointer, weight_pointer, bias_pointer, pre_act_pointer, output_pointer,
    batch_dim, in_feat_dim, out_feat_dim,
    input_batch_stride, input_in_feat_stride,
    weight_in_feat_stride, weight_out_feat_stride,
    add_bias: tl.constexpr, act_func: tl.constexpr, save_pre_act: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr, GROUP_SIZE_BATCH: tl.constexpr,
    ):
    """
    Linearly transforms the input using weights, optionally adding bias
    and fusing an activation function.

    Args:
        input_pointer: Pointer to the input to transform.
            The input must be of shape [batch_dim, in_feat_dim].
        weight_pointer: Pointer to the weights input is transformed by.
            The weights must be of shape [in_feat_dim, out_feat_dim].
        bias_pointer: Pointer to an optional additive bias vector.
            The bias vector, if provided, must be of shape [out_feat_dim].
        pre_act_pointer: Pointer to an optional container the pre-activation input
            is written to if act_func is not None and save_pre_act is True.
            The container, if provided, must be of shape [batch_dim, out_feat_dim]
            and contiguous.
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, out_feat_dim]
            and contiguous.
        batch_dim: Batch dimension of the input and output.
        in_feat_dim: Dimensionality of the input features.
        out_feat_dim: Dimensionality of the output features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_in_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        weight_in_feat_stride: Stride necessary to jump one element along the
            weights' input feature dimension.
        weight_out_feat_stride: Stride necessary to jump one element along the
            weights' output feature dimension.
        add_bias: Flag for adding a bias vector.
        act_func: Name of activation function to apply, with None for identity.
                    Options are 'sigmoid', 'tanh', 'relu', and 'gelu'.
        save_pre_act: Flag for saving the pre-activation input.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
        GROUP_SIZE_BATCH: Group size across the batch dimension.
    """
    # Programs are blocked together, GROUP_SIZE_BATCH rows at a time,
    # to alleviate L2 miss rates.
    pid = tl.program_id(axis=0)
    n_batch_pids = tl.cdiv(batch_dim, BLOCK_SIZE_BATCH)
    n_out_feat_pids = tl.cdiv(out_feat_dim, BLOCK_SIZE_OUT_FEAT)
    pids_per_group = GROUP_SIZE_BATCH * n_out_feat_pids
    group_id = pid // pids_per_group
    first_batch_pid = group_id * GROUP_SIZE_BATCH
    GROUP_SIZE_BATCH = min(n_batch_pids - first_batch_pid, GROUP_SIZE_BATCH)
    batch_pid = first_batch_pid + (pid % GROUP_SIZE_BATCH)
    out_feat_pid = (pid % pids_per_group) // GROUP_SIZE_BATCH

    batch_offset = (batch_pid * BLOCK_SIZE_BATCH +
                    tl.arange(0, BLOCK_SIZE_BATCH))
    out_feat_offset = (out_feat_pid * BLOCK_SIZE_OUT_FEAT +
                       tl.arange(0, BLOCK_SIZE_OUT_FEAT))

    batch_mask = batch_offset < batch_dim
    out_feat_mask = out_feat_offset < out_feat_dim

    input_pointer += input_batch_stride * batch_offset[:, None]
    weight_pointer += weight_out_feat_stride * out_feat_offset[None, :]

    accum = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_FEAT),
                     dtype=tl.float32)

    for block_ind in range(0, tl.cdiv(in_feat_dim, BLOCK_SIZE_IN_FEAT)):
        in_feat_offset = (block_ind * BLOCK_SIZE_IN_FEAT +
                          tl.arange(0, BLOCK_SIZE_IN_FEAT))
        in_feat_mask = in_feat_offset < in_feat_dim

        curr_input_pointer = (input_pointer +
                              input_in_feat_stride * in_feat_offset[None, :])
        curr_weight_pointer = (weight_pointer +
                               weight_in_feat_stride * in_feat_offset[:, None])

        input_block = tl.load(curr_input_pointer,
                              mask=batch_mask[:, None] & in_feat_mask[None, :])
        weight_block = tl.load(curr_weight_pointer,
                               mask=out_feat_mask[None, :] & in_feat_mask[:, None])

        accum += tl.dot(input_block, weight_block)

    if add_bias:
        bias = tl.load(bias_pointer + out_feat_offset,
                       mask=out_feat_mask)
        accum += bias[None, :]

    if act_func is not None:
        if save_pre_act:
            pre_act_pointer += (out_feat_dim * batch_offset[:, None] +
                                out_feat_offset[None, :])
            tl.store(pre_act_pointer, accum,
                     mask=batch_mask[:, None] & out_feat_mask[None, :])

        accum = apply_act_func(accum, act_func)

    output_pointer += (out_feat_dim * batch_offset[:, None] +
                       out_feat_offset[None, :])
    tl.store(output_pointer, accum,
             mask=batch_mask[:, None] & out_feat_mask[None, :])

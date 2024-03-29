"""
Batch normalization with residual addition and fused activation
with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .act_kernels import act_func_backward_kernel
from .batch_norm_kernels import batch_norm_backward_kernel, batch_norm_forward_kernel
from .types import Context, Device


def make_3d_for_bn(input: Tensor) -> Tensor:
    """
    Converts the input to a 3D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)

    elif input.ndim == 4:
        input = input.flatten(2, -1)

    return input


class BatchNormAutoGrad(torch.autograd.Function):
    """
    Autodiff for batch normalization.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        training: bool,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
        track_running_stats: bool = True,
        pre_act_add: Optional[Tensor] = None,
        act_func: Optional[str] = None,
        ) -> Tensor:
        """
        Batch-normalizes the input, optionally adding a residual and
        fusing an activation function.

        Args:
            ctx: Context for variable storage.
            input: Input to layer-normalize.
                Must be of shape [batch_dim, feat_dim] or [batch_dim, feat_dim, spatial_dim].
            training: Flag indicating if the model is in training mode.
            weight: Optional weights for affine transform when bias is provided.
                If provided, must be of shape [feat_dim].
            bias: Optional bias vector for affine transform when weight is provided.
                If provided, must be of shape [feat_dim].
            running_mean: Optional container for storing the input's running mean
                if training and track_running_stats are True.
            running_var: Optional container for storing the input's running variance
                if training and track_running_stats are True.
            momentum: Momentum for the running mean and variance.
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero.
            track_running_stats: Flag for tracking running mean and variance if
                is_train is also True.
            pre_act_add: Optional residual added to the pre-activation result.
            act_func: Name of activation function to apply, with None for identity.
                Options are 'sigmoid', 'tanh', 'relu', 'gelu', and 'silu'.

        Returns:
            Batch-normalized input, potentially with fused activation and added residual.
        """
        add_pre_act = pre_act_add is not None
        pre_act_add = (pre_act_add if add_pre_act else
                       torch.empty((1, 1, 1), device='cuda'))

        input_3d = make_3d_for_bn(input)
        pre_act_add = make_3d_for_bn(pre_act_add)
        transpose = False

        if input_3d.shape[-1] > 1:
            input_3d = input_3d.transpose(0, -1)
            pre_act_add = pre_act_add.transpose(0, -1)
            transpose = True

        affine = weight is not None and bias is not None
        requires_grad = (input.requires_grad or
                         pre_act_add.requires_grad or
                         (affine and weight.requires_grad) or
                         (affine and bias.requires_grad))
        save_pre_act = requires_grad and (act_func is not None)

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        output = torch.empty_like(input_3d)
        pre_act = torch.empty_like(input_3d) if save_pre_act else output

        if requires_grad:
            mean = torch.empty(feat_dim,
                               device=input.device,
                               dtype=torch.float32)
            inv_std = torch.empty(feat_dim,
                                  device=input.device,
                                  dtype=torch.float32)

        else:
            mean = inv_std = None

        running_mean = input if running_mean is None else running_mean
        running_var = input if running_var is None else running_var

        # Launches 1D grid where each program operates over one feature.
        grid = lambda _: (feat_dim,)
        batch_norm_forward_kernel[grid](input_3d, weight, bias,
                                        mean, inv_std,
                                        pre_act_add, pre_act, output,
                                        running_mean, running_var,
                                        batch_dim, spatial_dim,
                                        *input_3d.stride(), *pre_act_add.stride(),
                                        *pre_act.stride(), *output.stride(),
                                        momentum, eps,
                                        affine=affine,
                                        save_stats=requires_grad,
                                        track_running_stats=track_running_stats,
                                        is_train=training,
                                        add_pre_act=add_pre_act,
                                        act_func=act_func,
                                        save_pre_act=save_pre_act)

        if transpose:
            output = output.transpose(0, -1)
            if save_pre_act:
                pre_act = pre_act.transpose(0, -1)

        ctx.affine = affine
        ctx.act_func = act_func
        ctx.add_pre_act = add_pre_act
        if requires_grad:
            ctx.save_for_backward(input, mean, inv_std, weight,
                                  pre_act if save_pre_act else None)

        return output.view_as(input)

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of batch normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of batch normalization.
        """
        (input, mean, inv_std, weight, pre_act) = ctx.saved_tensors
        input_3d = make_3d_for_bn(input)

        if ctx.act_func is None:
            pre_act_grad = make_3d_for_bn(output_grad)

        else:
            size = output_grad.numel()
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype,
                                       device=pre_act.device)

            # Launches 1D grid where each program operates over
            # BLOCK_SIZE elements.
            grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
            act_func_backward_kernel[grid](output_grad.flatten(), pre_act,
                                           pre_act_grad, size, ctx.act_func)

            pre_act_grad = pre_act_grad.view_as(pre_act)

        transpose = False
        if input_3d.shape[-1] > 1:
            input_3d = input_3d.transpose(0, -1)
            pre_act_grad = pre_act_grad.transpose(0, -1)
            transpose = True

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        input_grad = torch.empty_like(input_3d)

        if ctx.affine:
            weight_grad = torch.empty((feat_dim,), device=input.device)
            bias_grad = torch.empty_like(weight_grad)

        else:
            weight_grad = bias_grad = None

        # Launches 1D grid where each program operates over one feature.
        grid = lambda _: (feat_dim,)
        batch_norm_backward_kernel[grid](pre_act_grad, input_3d, mean, inv_std, weight,
                                         input_grad, weight_grad, bias_grad,
                                         batch_dim, spatial_dim,
                                         *pre_act_grad.stride(),
                                         *input_3d.stride(), *input_grad.stride(),
                                         affine=ctx.affine)

        if transpose:
            input_grad = input_grad.transpose(0, -1)
            pre_act_grad = pre_act_grad.transpose(0, -1)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return (input_grad.view_as(input), None,
                weight_grad, bias_grad,
                None, None, None, None, None,
                pre_act_grad.view_as(input) if ctx.add_pre_act else None,
                None)


class BatchNorm1d(nn.BatchNorm1d):
    """
    Batch-normalizes the 2D or 3D input, optionally fusing an activation function
    and adding a residual to the pre-activation result.
    See also base class.

    Args:
        num_features: Number of features.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        momentum: Momentum for the running mean and variance.
        affine: Flag for performing an affine transformation on the normalized output.
        track_running_stats: Flag for tracking running mean and variance if
            is_train is also True.
        act_func: Name of activation function to apply, with None for identity.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', and 'silu'.
        device: Device to use.
        dtype: Dtype of layer.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        act_func: Optional[str] = None,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats, device, dtype)
        self.act_func = act_func

    def forward(
        self,
        input: Tensor,
        pre_act_add: Optional[Tensor] = None,
        ) -> Tensor:
        self._check_input_dim(input)

        return BatchNormAutoGrad.apply(input, self.training,
                                       self.weight, self.bias,
                                       self.running_mean, self.running_var,
                                       self.momentum, self.eps,
                                       self.track_running_stats,
                                       pre_act_add, self.act_func)


class BatchNorm2d(nn.BatchNorm2d):
    """
    Batch-normalizes the 4D input, optionally adding a residual
    and fusing an activation function.
    See also base class.

    Args:
        num_features: Number of features.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        momentum: Momentum for the running mean and variance.
        affine: Flag for performing an affine transformation on the normalized output.
        track_running_stats: Flag for tracking running mean and variance if
            is_train is also True.
        act_func: Name of activation function to apply, with None for identity.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', and 'silu'.
        device: Device to use.
        dtype: Dtype of layer.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        act_func: Optional[str] = None,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats, device, dtype)
        self.act_func = act_func

    def forward(
        self,
        input: Tensor,
        pre_act_add: Optional[Tensor] = None,
        ) -> Tensor:
        self._check_input_dim(input)

        return BatchNormAutoGrad.apply(input, self.training,
                                       self.weight, self.bias,
                                       self.running_mean, self.running_var,
                                       self.momentum, self.eps,
                                       self.track_running_stats,
                                       pre_act_add, self.act_func)

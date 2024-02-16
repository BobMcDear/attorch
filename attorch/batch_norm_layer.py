"""
Batch normalization with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn

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
        ) -> Tensor:
        """
        Batch-normalizes the input.

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

        Returns:
            Batch-normalized input.
        """
        input_3d = make_3d_for_bn(input)
        transpose = False

        if input_3d.shape[-1] > 1:
            input_3d = input_3d.transpose(0, -1)
            transpose = True

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        output = torch.empty_like(input_3d)

        affine = weight is not None and bias is not None
        requires_grad = (input.requires_grad or
                         (affine and weight.requires_grad) or
                         (affine and bias.requires_grad))

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
                                        mean, inv_std, output,
                                        running_mean, running_var,
                                        batch_dim, spatial_dim,
                                        *input_3d.stride(), *output.stride(),
                                        momentum, eps,
                                        affine=affine,
                                        save_stats=requires_grad,
                                        track_running_stats=track_running_stats,
                                        is_train=training)

        if transpose:
            output = output.transpose(0, -1)

        ctx.affine = affine
        if requires_grad:
            ctx.save_for_backward(input, mean, inv_std, weight)

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
        (input, mean, inv_std, weight) = ctx.saved_tensors
        input_3d = make_3d_for_bn(input)
        output_grad_3d = make_3d_for_bn(output_grad)

        transpose = False
        if input_3d.shape[-1] > 1:
            input_3d = input_3d.transpose(0, -1)
            output_grad_3d = output_grad_3d.transpose(0, -1)
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
        batch_norm_backward_kernel[grid](output_grad_3d, input_3d, mean, inv_std, weight,
                                         input_grad, weight_grad, bias_grad,
                                         batch_dim, spatial_dim,
                                         *output_grad_3d.stride(),
                                         *input_3d.stride(), *input_grad.stride(),
                                         affine=ctx.affine)

        if transpose:
            input_grad = input_grad.transpose(0, -1)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return (input_grad.view_as(input), None,
                weight_grad, bias_grad,
                None, None, None, None, None)


class BatchNorm1d(nn.BatchNorm1d):
    """
    Batch-normalizes the 2D or 3D input.
    See also base class.

    Args:
        num_features: Number of features.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        momentum: Momentum for the running mean and variance.
        affine: Flag for performing an affine transformation on the normalized output.
        track_running_stats: Flag for tracking running mean and variance if
            is_train is also True.
        device: Device to use. Only CUDA is supported.
        dtype: Dtype of layer. Only float32 is supported.

    Raises:
        RuntimeError: 1. A device other than CUDA was passed.
                      2. A dtype other than float32 was passed.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        if 'cuda' not in str(device):
            raise RuntimeError('Batch normalization only supports CUDA devices.')

        if dtype is not torch.float32:
            raise RuntimeError('Batch normalization only supports float32 dtype.')

        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        return BatchNormAutoGrad.apply(input, self.training,
                                       self.weight, self.bias,
                                       self.running_mean, self.running_var,
                                       self.momentum, self.eps,
                                       self.track_running_stats)


class BatchNorm2d(nn.BatchNorm2d):
    """
    Batch-normalizes the 4D input.
    See also base class.

    Args:
        num_features: Number of features.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        momentum: Momentum for the running mean and variance.
        affine: Flag for performing an affine transformation on the normalized output.
        track_running_stats: Flag for tracking running mean and variance if
            is_train is also True.
        device: Device to use. Only CUDA is supported.
        dtype: Dtype of layer. Only float32 is supported.

    Raises:
        RuntimeError: 1. A device other than CUDA was passed.
                      2. A dtype other than float32 was passed.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        if 'cuda' not in str(device):
            raise RuntimeError('Batch normalization only supports CUDA devices.')

        if dtype is not torch.float32:
            raise RuntimeError('Batch normalization only supports float32 dtype.')

        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        return BatchNormAutoGrad.apply(input, self.training, self.weight, self.bias,
                                       self.running_mean, self.running_var,
                                       self.momentum, self.eps,
                                       self.track_running_stats)

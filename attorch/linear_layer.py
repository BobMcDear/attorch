"""
Linear layer with fused activation with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .act_kernels import act_func_backward_kernel
from .linear_kernels import linear_forward_kernel
from .types import Context, Device


class LinearAutoGrad(torch.autograd.Function):
    """
    Autodiff for linear layer.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        act_func: Optional[str] = None,
        ) -> Tensor:
        """
        Linearly transforms the input using weights, optionally adding bias
        and fusing an activation function.

        Args:
            input: Input to transform.
                Must be of shape [..., in_feat_dim].
            weight: Weights input is transformed by.
                Must be of shape [in_feat_dim, out_feat_dim].
            bias: Optional additive bias vector, with None for no bias.
                If provided, must be of shape [out_feat_dim].
            act_func: Name of activation function to apply, with None for identity.
                    Options are 'sigmoid', 'tanh', 'relu', and 'gelu'.

        Returns:
            Input linearly transformed, potentially with added biased and
            fused activation.
        """
        assert weight.ndim == 2, \
            f'Weights must be 2D, received shape {weight.shape}'
        assert bias is None or bias.ndim == 1, \
            f'Bias must be 1D, received shape {bias.shape}'

        assert input.shape[-1] == weight.shape[0], \
            f'Incompatible input ({input.shape}) and weights ({weight.shape}) shape'
        assert bias is None or weight.shape[1] == bias.shape[0], \
            f'Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape'

        flattened_input = input.flatten(0, -2)
        batch_dim, in_feat_dim = flattened_input.shape
        _, out_feat_dim = weight.shape

        requires_grad = (input.requires_grad or
                         weight.requires_grad or
                         (bias is not None and bias.requires_grad))
        save_pre_act = requires_grad and act_func is not None

        output = torch.empty((batch_dim, out_feat_dim),
                             device=input.device,
                             dtype=input.dtype)
        pre_act = torch.empty_like(output) if save_pre_act else input

        # Launches a 1D grid, where each program outputs blocks of
        # BLOCK_SIZE_BATCH rows and BLOCK_SIZE_OUT_FEAT columns.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']) *
                             cdiv(out_feat_dim, META['BLOCK_SIZE_OUT_FEAT']),)
        linear_forward_kernel[grid](flattened_input, weight,
                                    input if bias is None else bias,
                                    pre_act, output,
                                    batch_dim, in_feat_dim, out_feat_dim,
                                    *flattened_input.stride(), *weight.stride(),
                                    add_bias=bias is not None, act_func=act_func,
                                    save_pre_act=save_pre_act)


        ctx.act_func = act_func
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        if requires_grad:
            ctx.save_for_backward(input, pre_act if save_pre_act else None, weight)

        return output.view(*input.shape[:-1], out_feat_dim)

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of the linear layer.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the linear layer.
        """
        input, pre_act, weight = ctx.saved_tensors

        output_grad = output_grad.flatten(0, -2)
        flattened_input = input.flatten(0, -2)

        batch_dim, _ = flattened_input.shape
        _, out_feat_dim = weight.shape

        if ctx.act_func is None:
            pre_act_grad = output_grad

        else:
            size = batch_dim * out_feat_dim
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype,
                                       device=pre_act.device)

            # Launches 1D grid where each program operates over
            # BLOCK_SIZE elements.
            grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
            act_func_backward_kernel[grid](output_grad, pre_act, pre_act_grad,
                                           size, ctx.act_func)

            pre_act_grad = pre_act_grad.view_as(pre_act)

        # Using PyTorch's matmul, but linear_forward_kernel
        # could have also been used.
        input_grad = pre_act_grad @ weight.T if input.requires_grad else None
        weight_grad = (flattened_input.T @ pre_act_grad
                       if weight.requires_grad else None)
        bias_grad = pre_act_grad.sum(dim=0) if ctx.bias_requires_grad else None

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return (input_grad.view_as(input) if input_grad is not None else None,
                weight_grad, bias_grad, None)


class Linear(nn.Linear):
    """
    Linearly transforms the input using weights, optionally adding bias
    and fusing an activation function.
    See also base class.

    Note: Unlike PyTorch's linear layer, the weight matrix in this module is
    of shape [in_features, out_features] instead of [out_features, in_features].
    This may cause unexpected issues when manipulating the weights (e.g., porting
    parameters, initializing them, and so forth).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Flag for additive bias.
        act_func: Name of activation function to apply, with None for identity.
            Options are 'sigmoid', 'tanh', 'relu', and 'gelu'.
        device: Device to use. Only CUDA is supported.
        dtype: Dtype of layer. Only float32 is supported.

    Raises:
        RuntimeError: 1. A device other than CUDA was passed.
                      2. A dtype other than float32 was passed.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func: Optional[str] = None,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        if 'cuda' not in str(device):
            raise RuntimeError('Linear layer only supports CUDA devices.')

        if dtype is not torch.float32:
            raise RuntimeError('Linear layer only supports float32 dtype.')

        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight = nn.Parameter(self.weight.T.contiguous())
        self.act_func = act_func

    def forward(self, input: Tensor) -> Tensor:
        return LinearAutoGrad.apply(input, self.weight, self.bias,
                                    self.act_func)

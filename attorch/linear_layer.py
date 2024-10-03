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
from .utils import get_output_dtype


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
                Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
                'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and
                'leaky_relu_PARAM', where PARAM stands for the parameter in the
                case of parameterized activation functions (e.g., 'leaky_relu_0.01'
                for leaky ReLU with a negative slope of 0.01).

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

        param = None
        if '_' in act_func:
            comps = act_func.split('_')
            act_func = '_'.join(comps[:-1])
            param = float(comps[-1])

        flattened_input = input.flatten(0, -2)
        batch_dim, in_feat_dim = flattened_input.shape
        _, out_feat_dim = weight.shape

        requires_grad = (input.requires_grad or
                         weight.requires_grad or
                         (bias is not None and bias.requires_grad))
        save_pre_act = requires_grad and (act_func is not None)

        output_dtype = get_output_dtype(input.dtype, autocast='fp16')
        output = torch.empty((batch_dim, out_feat_dim),
                             device=input.device,
                             dtype=output_dtype)
        pre_act = torch.empty_like(output) if save_pre_act else output

        # Launches a 1D grid, where each program outputs blocks of
        # BLOCK_SIZE_BATCH rows and BLOCK_SIZE_OUT_FEAT columns.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']) *
                             cdiv(out_feat_dim, META['BLOCK_SIZE_OUT_FEAT']),)
        linear_forward_kernel[grid](flattened_input, weight,
                                    input if bias is None else bias,
                                    pre_act, output,
                                    batch_dim, in_feat_dim, out_feat_dim,
                                    *flattened_input.stride(), *weight.stride(),
                                    *pre_act.stride(), *output.stride(), param,
                                    add_bias=bias is not None, act_func=act_func,
                                    save_pre_act=save_pre_act,
                                    fp16=output_dtype is torch.float16)

        ctx.param = param
        ctx.act_func = act_func
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_dtype = output_dtype
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
                                           size, None, None, ctx.param,
                                           ctx.act_func, False)

            pre_act_grad = pre_act_grad.view_as(pre_act)

        # Using PyTorch's matmul, but linear_forward_kernel
        # could have also been used.
        with torch.autocast('cuda', dtype=ctx.output_dtype):
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
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and
            'leaky_relu_PARAM', where PARAM stands for the parameter in the
            case of parameterized activation functions (e.g., 'leaky_relu_0.01'
            for leaky ReLU with a negative slope of 0.01).
        device: Device to use.
        dtype: Dtype of layer.
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
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight = nn.Parameter(self.weight.T.contiguous())
        self.act_func = act_func

    def forward(self, input: Tensor) -> Tensor:
        return LinearAutoGrad.apply(input, self.weight, self.bias,
                                    self.act_func)

"""
Gated linear unit layer with arbitrary activation functions
with PyTorch autodiff support.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.amp import custom_bwd, custom_fwd
from triton import cdiv

from .glu_kernels import glu_backward_kernel, glu_forward_kernel
from .types import Context


class GLUAutoGrad(torch.autograd.Function):
    """
    Autodiff for gated linear unit.
    """
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(
        ctx: Context,
        input: Tensor,
        dim: int,
        act_func: str,
        ) -> Tensor:
        """
        Applies the gated linear unit with an arbitrary activation function
        to the input.

        Args:
            ctx: Context for variable storage.
            input: Input to gate.
                Can have arbitrary shape but dimension dim must be even.
            dim: Dimension over which to gate.
            act_func: Name of activation function to apply.
                Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
                'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and
                'leaky_relu_PARAM', where PARAM stands for the parameter in the
                case of parameterized activation functions (e.g., 'leaky_relu_0.01'
                for leaky ReLU with a negative slope of 0.01).

        Returns:
            Input transformed by the gated linear unit
            with an arbitrary activation function.
        """
        param = None
        if '_' in act_func:
            comps = act_func.split('_')
            act_func = '_'.join(comps[:-1])
            param = float(comps[-1])

        input1, input2 = input.chunk(2, dim=dim)
        input1 = input1.contiguous()
        input2 = input2.contiguous()

        requires_grad = input.requires_grad
        size = input1.numel()
        output = torch.empty_like(input1)

        ctx.param = param
        ctx.act_func = act_func
        ctx.dim = dim
        ctx.size = size
        if requires_grad:
            ctx.save_for_backward(input1, input2)

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements.
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        glu_forward_kernel[grid](input1, input2, output, size, param, act_func)

        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of the gated linear unit.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the gated linear unit.
        """
        (input1, input2) = ctx.saved_tensors
        input1_grad = torch.empty_like(input1)
        input2_grad = torch.empty_like(input2)

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements.
        grid = lambda META: (cdiv(ctx.size, META['BLOCK_SIZE']),)
        glu_backward_kernel[grid](output_grad, input1, input2,
                                  input1_grad, input2_grad,
                                  ctx.size, ctx.param, ctx.act_func)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return torch.concat([input1_grad, input2_grad], dim=ctx.dim), None, None


class GLU(nn.GLU):
    """
    Applies the gated linear unit with an arbitrary activation function
    to the input.
    See also base class.

    Args:
        dim: Dimension over which to gate.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and
            'leaky_relu_PARAM', where PARAM stands for the parameter in the
            case of parameterized activation functions (e.g., 'leaky_relu_0.01'
            for leaky ReLU with a negative slope of 0.01).
    """
    def __init__(self, dim: int = -1, act_func: str = 'sigmoid') -> None:
        super().__init__(dim)
        self.act_func = act_func

    def forward(self, input: Tensor) -> Tensor:
        return GLUAutoGrad.apply(input, self.dim, self.act_func)

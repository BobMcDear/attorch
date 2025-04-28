"""
Activation function layers with fused dropout with PyTorch autodiff support.
"""


import warnings
from random import randint
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.amp import custom_bwd, custom_fwd
from triton import cdiv

from .act_kernels import act_func_backward_kernel, act_func_forward_kernel
from .types import Context


class ActFuncAutoGrad(torch.autograd.Function):
    """
    Autodiff for activation functions.
    """
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(
        ctx: Context,
        input: Tensor,
        act_func: str,
        drop_p: float,
        training: bool,
        ) -> Tensor:
        """
        Applies an activation function to the input, optionally fusing dropout.

        Args:
            ctx: Context for variable storage.
            input: Input to transform.
                Can have arbitrary shape.
            act_func: Name of activation function to apply.
                Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'silu',
                'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
                'softplus', 'softsign', 'tanhshrink', 'leaky_relu_PARAM',
                'elu_PARAM', 'celu_PARAM', and 'hardshrink_PARAM' where PARAM stands for the parameter in the
                case of parameterized activation functions (e.g., 'leaky_relu_0.01'
                for leaky ReLU with a negative slope of 0.01).
            drop_p: Probability of dropping an element for dropout.
            training: Flag indicating if the model is in training mode,
                only applicable if drop_p is greater than 0.
                If False, no dropout is applied.

        Returns:
            Input transformed by the desired activation function,
            potentially with fused dropout.
        """
        param = None
        if '_' in act_func:
            comps = act_func.split('_')
            act_func = '_'.join(comps[:-1])
            param = float(comps[-1])

        ctx.param = param
        ctx.act_func = act_func
        ctx.drop_p = drop_p
        ctx.dropout = drop_p > 0 and training
        seed = randint(0, 65535) if ctx.dropout else 0
        ctx.seed = seed
        if input.requires_grad:
            ctx.save_for_backward(input)

        flattened_input = input.flatten()
        size = len(flattened_input)
        output = torch.empty_like(flattened_input)

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements.
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        act_func_forward_kernel[grid](flattened_input, output, size,
                                      drop_p, seed, param,
                                      act_func, ctx.dropout)

        return output.view_as(input)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of the activation function.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the activation function.
        """
        (input,) = ctx.saved_tensors
        flattened_input = input.flatten()
        output_grad = output_grad.flatten()

        size = len(flattened_input)
        input_grad = torch.empty_like(flattened_input)

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements.
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        act_func_backward_kernel[grid](output_grad, flattened_input, input_grad,
                                       size, ctx.drop_p, ctx.seed, ctx.param,
                                       ctx.act_func, ctx.dropout)

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return input_grad.view_as(input), None, None, None


class Sigmoid(nn.Sigmoid):
    """
    Applies sigmoid to the input, optionally fusing dropout.
    See also base class.

    Args:
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, drop_p: float = 0.0) -> None:
        super().__init__()
        self.drop_p = drop_p

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'sigmoid', self.drop_p, self.training)


class LogSigmoid(nn.LogSigmoid):
    """
    Applies the log of sigmoid to the input, optionally fusing dropout.
    See also base class.

    Args:
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, drop_p: float = 0.0) -> None:
        super().__init__()
        self.drop_p = drop_p

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'logsigmoid', self.drop_p, self.training)


class Tanh(nn.Tanh):
    """
    Applies tanh to the input, optionally fusing dropout.
    See also base class.

    Args:
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, drop_p: float = 0.0) -> None:
        super().__init__()
        self.drop_p = drop_p

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'tanh', self.drop_p, self.training)


class ReLU(nn.ReLU):
    """
    Applies ReLU to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, inplace: bool = False, drop_p: float = 0.0) -> None:
        super().__init__(inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place ReLU currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'relu', self.drop_p, self.training)


class GELU(nn.GELU):
    """
    Applies GELU to the input, optionally fusing dropout.
    See also base class.

    Args:
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, drop_p: float = 0.0) -> None:
        super().__init__()
        self.drop_p = drop_p

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'gelu', self.drop_p, self.training)


class SiLU(nn.SiLU):
    """
    Applies SiLU to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, inplace: bool = False, drop_p: float = 0.0) -> None:
        super().__init__(inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place SiLU currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'silu', self.drop_p, self.training)


class ReLU6(nn.ReLU6):
    """
    Applies ReLU6 to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, inplace: bool = False, drop_p: float = 0.0) -> None:
        super().__init__(inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place ReLU6 currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'relu6', self.drop_p, self.training)


class Hardsigmoid(nn.Hardsigmoid):
    """
    Applies hard sigmoid to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, inplace: bool = False, drop_p: float = 0.0) -> None:
        super().__init__(inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place hard sigmoid currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'hardsigmoid', self.drop_p, self.training)


class Hardtanh(nn.Hardtanh):
    """
    Applies hard tanh to the input, optionally fusing dropout.
    See also base class.

    Args:
        min_val: This argument is not supported.
        max_val: This argument is not supported.
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        drop_p: Probability of dropping an element for dropout.

    Raises:
        RuntimeError: 1. Minimum value was not set to -1.
                      2. Maximum value was not set to 1.
    """
    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        drop_p: float = 0.0,
        ) -> None:
        if min_val != -1.0:
            raise RuntimeError('Hard tanh only supports a minimum value of -1.')

        if max_val != 1.0:
            raise RuntimeError('Hard tanh only supports a maximum value of -1.')

        super().__init__(inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place hard sigmoid currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'hardtanh', self.drop_p, self.training)


class Hardswish(nn.Hardswish):
    """
    Applies hard Swish to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, inplace: bool = False, drop_p: float = 0.0) -> None:
        super().__init__(inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place hard Swish currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'hardswish', self.drop_p, self.training)


class SELU(nn.SELU):
    """
    Applies SELU to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, inplace: bool = False, drop_p: float = 0.0) -> None:
        super().__init__(inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place SELU currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'selu', self.drop_p, self.training)


class Mish(nn.Mish):
    """
    Applies Mish to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, inplace: bool = False, drop_p: float = 0.0) -> None:
        super().__init__(inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place Mish currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'mish', self.drop_p, self.training)


class Softplus(nn.Softplus):
    """
    Applies softplus to the input, optionally fusing dropout.
    See also base class.

    Args:
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, drop_p: float = 0.0) -> None:
        super().__init__()
        self.drop_p = drop_p

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'softplus', self.drop_p, self.training)


class Softsign(nn.Softsign):
    """
    Applies softsign to the input, optionally fusing dropout.
    See also base class.

    Args:
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, drop_p: float = 0.0) -> None:
        super().__init__()
        self.drop_p = drop_p

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'softsign', self.drop_p, self.training)


class Tanhshrink(nn.Tanhshrink):
    """
    Applies tanhshrink to the input, optionally fusing dropout.
    See also base class.

    Args:
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(self, drop_p: float = 0.0) -> None:
        super().__init__()
        self.drop_p = drop_p

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input, 'tanhshrink', self.drop_p, self.training)


class LeakyReLU(nn.LeakyReLU):
    """
    Applies leaky ReLU to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        negative_slope: Slope of the negative component.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(
        self,
        inplace: bool = False,
        negative_slope: float = 1e-2,
        drop_p: float = 0.0,
        ) -> None:
        super().__init__(inplace=False)
        self.negative_slope = negative_slope
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place leaky ReLU currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input,
                                     'leaky_relu_' + str(self.negative_slope),
                                     self.drop_p, self.training)


class ELU(nn.ELU):
    """
    Applies ELU to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        alpha: Alpha value.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(
        self,
        inplace: bool = False,
        alpha: float = 1.0,
        drop_p: float = 0.0,
        ) -> None:
        super().__init__(alpha, inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place ELU currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input,
                                     'elu_' + str(self.alpha),
                                     self.drop_p, self.training)


class CELU(nn.CELU):
    """
    Applies CELU to the input, optionally fusing dropout.
    See also base class.

    Args:
        inplace: This is a dummy argument and has no effects,
            as in-place is currently not supported.
        alpha: Alpha value.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(
        self,
        inplace: bool = False,
        alpha: float = 1.0,
        drop_p: float = 0.0,
        ) -> None:
        super().__init__(alpha, inplace=False)
        self.drop_p = drop_p

        if inplace is True:
            warnings.warn('In-place CELU currently not supported; '
                          'falling back to out-of-place.')

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input,
                                     'celu_' + str(self.alpha),
                                     self.drop_p, self.training)


class Hardshrink(nn.Hardshrink):
    """
    Applies hard shrink to the input, optionally fusing dropout.
    See also base class.

    Args:
        lambd: Lambda value.
        drop_p: Probability of dropping an element for dropout.
    """
    def __init__(
        self,
        lambd: float = 0.5,
        drop_p: float = 0.0,
        ) -> None:
        super().__init__(lambd)
        self.drop_p = drop_p

    def forward(self, input: Tensor) -> Tensor:
        return ActFuncAutoGrad.apply(input,
                                     'hardshrink_' + str(self.lambd),
                                     self.drop_p, self.training)

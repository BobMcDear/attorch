"""
Pooling layers with PyTorch autodiff support.
"""


from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.utils import _pair

from .conv_layer import Conv2dAutoGrad


class AvgPool2d(nn.AvgPool2d):
    """
    Averages 2D windows of pixels in the input.
    See also base class.

    Note: The Triton compiler does not perform well with convolutional kernels,
    which underlie this pooling layer, and a significant speed disparity between
    this module and its PyTorch equivalent should be expected. Use at your own discretion.

    Args:
        kernel_size: Kernel size.
            If an int, this value is used along both spatial dimensions.
        stride: Stride of kernel.
            If an int, this value is used along both spatial dimensions.
            If None, the kernel size is used as the stride.
        padding: Padding applied to the input.
            If an int, this value is used along both spatial dimensions.
        ceil_mode: Flag for using ceil instead of floor to calculate the output shape.
            Must be False.
        count_include_pad: Flag for including padding when averaging pixels.
            Must be True.
        divisor_override: Average divisor.
            Must be None.

    Raises:
        RuntimeError: 1. Ceil was requested in place of floor to calculate the otuput shape.
                      2. Padding was requested to be excluded in average computations.
                      3. Average divisor was overriden.
    """
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        ) -> None:
        if ceil_mode:
            raise RuntimeError('The output shape can only be calculated using floor, not ceil.')

        if not count_include_pad:
            raise RuntimeError('Padding must be included when averaging pixels.')

        if divisor_override is not None:
            raise RuntimeError('The average divisor must be the size of the window.')

        super().__init__(kernel_size, stride, padding, ceil_mode,
                         count_include_pad, divisor_override)

        self.kernel_size = _pair(self.kernel_size)
        self.stride = _pair(self.stride)
        self.padding = _pair(self.padding)
        self.kernel = None

    def forward(self, input: Tensor) -> Tensor:
        if self.kernel is None:
            self.kernel = torch.full((input.shape[1], 1, *self.kernel_size),
                                     1 / (self.kernel_size[0] * self.kernel_size[1]),
                                     device='cuda')

        return Conv2dAutoGrad.apply(input, self.kernel, None,
                                    *self.stride, *self.padding,
                                    input.shape[1])

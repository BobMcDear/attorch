"""
Convolutional layer with PyTorch autodiff support.
"""


from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .conv_kernels import conv2d_forward_kernel
from .types import Context, Device
from .utils import get_output_dtype


def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ) -> int:
    """
    Determines the output size of a 2D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.

    Returns:
        Output size of 2D convolution.
    """
    return (in_size + 2 * padding - kernel_size) // stride + 1


class Conv2dAutoGrad(torch.autograd.Function):
    """
    Autodiff for 2D convolutional layer.
    """
    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride_height: int = 1,
        stride_width: int = 1,
        padding_height: int = 1,
        padding_width: int = 1,
        groups: int = 1,
        ) -> Tensor:
        """
        2D-convolves over the input using weights, optionally adding bias.

        Args:
            input: Input to convolve over.
                Must be of shape [batch_dim, in_feat_dim, in_height, in_width].
            weight: Weights input is convolved over by.
                Must be of shape [out_feat_dim, in_feat_dim, kernel_height, kernel_width].
            bias: Optional additive bias vector, with None for no bias.
                If provided, must be of shape [out_feat_dim].
            stride_height: Stride of kernel across the height dimension.
            stride_width: Stride of kernel across the width dimension.
            padding_height: Padding applied to the input across the height dimension.
            padding_width: Padding applied to the input across the width dimension.
            groups: Number of groups for the convolution.

        Returns:
            Input 2D-convolved over, potentially with added biased.
        """
        assert weight.ndim == 4, \
            f'Weights must be 4D, received shape {weight.shape}'
        assert bias is None or bias.ndim == 1, \
            f'Bias must be 1D, received shape {bias.shape}'

        assert input.shape[1] == groups * weight.shape[1], \
            f'Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups'
        assert bias is None or weight.shape[0] == bias.shape[0], \
            f'Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape'

        batch_dim, in_feat_dim, in_height, in_width = input.shape
        out_feat_dim, _, kernel_height, kernel_width = weight.shape
        out_height = conv2d_output_size(in_height, kernel_height,
                                      stride_height, padding_height)
        out_width = conv2d_output_size(in_width, kernel_width,
                                     stride_width, padding_width)

        output_dtype = get_output_dtype(input.dtype, autocast='fp16')
        output = torch.empty((batch_dim, out_feat_dim, out_height, out_width),
                             device=input.device,
                             dtype=output_dtype)

        # Launches a 3D grid, where each program outputs blocks of
        # BLOCK_SIZE_BATCH_HEIGHT_WIDTH along the batch, height, and width dimensions,
        # BLOCK_SIZE_OUT_FEAT along the feature dimension, and one group.
        grid = lambda META: (cdiv(batch_dim * out_height * out_width, META['BLOCK_SIZE_BATCH_HEIGHT_WIDTH']),
                             cdiv(out_feat_dim, META['BLOCK_SIZE_OUT_FEAT']),
                             groups)
        conv2d_forward_kernel[grid](input, weight, output,
                                    batch_dim, in_feat_dim, in_height, in_width,
                                    out_feat_dim, out_height, out_width,
                                    *input.stride(),
                                    *weight.stride(),
                                    *output.stride(),
                                    kernel_height, kernel_width,
                                    stride_height, stride_width,
                                    padding_height, padding_width,
                                    groups=groups,
                                    fp16=output_dtype is torch.float16)

        if bias is not None:
            # Adding bias in the kernel becomes buggy when groups != 1.
            output += bias.view(1, -1, 1, 1)

        requires_grad = (input.requires_grad or
                         weight.requires_grad or
                         (bias is not None and bias.requires_grad))

        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.groups = groups
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_dtype = output_dtype
        if requires_grad:
            ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
        ) -> Tuple[Optional[Tensor], ...]:
        """
        Calculates the input gradient of the 2D convolutional layer.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the 2D convolutional layer.
        """
        input, weight = ctx.saved_tensors

        input = input.to(ctx.output_dtype)
        weight = weight.to(ctx.output_dtype)

        # The backward pass for the convolution operation requires dilation,
        # which is not supported by the kernel.
        input_grad = nn.grad.conv2d_input(input.shape, weight, output_grad,
                                          ctx.stride, ctx.padding,
                                          groups=ctx.groups)
        weight_grad = nn.grad.conv2d_weight(input, weight.shape, output_grad,
                                            ctx.stride, ctx.padding,
                                            groups=ctx.groups)
        bias_grad = (output_grad.sum(dim=(0, 2, 3)).to(ctx.output_dtype)
                     if ctx.bias_requires_grad else None)

        return input_grad, weight_grad, bias_grad, None, None, None, None, None


class Conv2d(nn.Conv2d):
    """
    2D-convolves over the input using weights, optionally adding bias.
    See also base class.

    Note: The Triton compiler does not perform well with convolutional kernels,
    and a significant speed disparity between this module and its PyTorch equivalent
    should be expected. Use at your own discretion.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size.
            If an int, this value is used along both spatial dimensions.
        stride: Stride of kernel.
            If an int, this value is used along both spatial dimensions.
        padding: Padding applied to the input.
            If an int, this value is used along both spatial dimensions.
        dilation: Dilation of kernel. Only 1 and (1, 1) are supported.
        groups: Number of groups for the convolution.
        bias: Flag for additive bias.
        padding_mode: Padding mode. Only 'zeros' is supported.
        device: Device to use. Only CUDA is supported.
        dtype: Dtype of layer. Only float32 is supported.

    Raises:
        RuntimeError: 1. A dilation other than 1 was passed.
                      2. A padding mode other than 'zeros' was passed.
                      3. A device other than CUDA was passed.
                      4. A dtype other than float32 was passed.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode, device, dtype)

        if self.dilation != (1, 1):
            raise RuntimeError('Convolutional layer only supports dilation of 1 and (1, 1).')

        if self.padding_mode != 'zeros':
            raise RuntimeError("Convolutional layer only support 'zeros' padding mode.")

        if 'cuda' not in str(device):
            raise RuntimeError('Convolutional layer only supports CUDA devices.')

        if dtype is not torch.float32:
            raise RuntimeError('Convolutional layer only supports float32 dtype.')


    def forward(self, input: Tensor) -> Tensor:
        return Conv2dAutoGrad.apply(input, self.weight, self.bias,
                                    *self.stride, *self.padding,
                                    self.groups)

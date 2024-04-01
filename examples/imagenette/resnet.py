"""
ResNet for classification.
"""


from typing import Optional, Tuple

from torch import Tensor
from torch import nn

import attorch


class ConvBNReLU(nn.Module):
    """
    Transforms the input using a convolution, followed by batch normalization
    and possible ReLU. A residual can optionally be added to the pre-activation result.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        in_dim: Number of input channels.
        out_dim: Number of output channels of the convolution.
        kernel_size: Kernel size of the convolution.
        stride: Stride of the convolution.
        padding: Padding of the convolution.
            If None, it is set to half the kernel size, rounded down.
        relu: Flag for appending ReLU after batch normalization.
    """
    def __init__(
        self,
        use_attorch: bool,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        relu: bool = True,
        ) -> None:
        super().__init__()
        self.use_attorch = use_attorch

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size,
                              bias=False, stride=stride,
                              padding=padding or kernel_size // 2)

        if use_attorch:
            self.bn = attorch.BatchNorm2d(out_dim, act_func='relu' if relu else None)

        else:
            self.bn = nn.BatchNorm2d(out_dim)
            self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(
        self,
        input: Tensor,
        pre_act_add: Optional[Tensor] = None,
        ) -> Tensor:
        output = self.conv(input)

        if self.use_attorch:
            output = self.bn(output, pre_act_add)

        else:
            output = self.bn(output)
            if pre_act_add is not None:
                output = output + pre_act_add
            output = self.relu(output)

        return output


class BottleneckBlock(nn.Module):
    """
    Passes the input through a residual bottleneck block.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        in_dim: Number of input channels.
        bottleneck_dim: Number of bottleneck channels.
        stride: Stride of the middle convolution.
        expansion_factor: Factor by which the bottleneck dimension is expanded
            to generate the output.
    """
    def __init__(
        self,
        use_attorch: bool,
        in_dim: int,
        bottleneck_dim: int,
        stride: int = 1,
        expansion_factor: int = 4,
        ) -> None:
        super().__init__()

        out_dim = expansion_factor * bottleneck_dim
        self.conv_bn_relu1 = ConvBNReLU(use_attorch, in_dim, bottleneck_dim)
        self.conv_bn_relu2 = ConvBNReLU(use_attorch, bottleneck_dim, bottleneck_dim,
                                        kernel_size=3, stride=stride)
        self.conv_bn_relu3 = ConvBNReLU(use_attorch, bottleneck_dim, out_dim)

        self.downsample = nn.Identity()
        if in_dim != out_dim or stride != 1:
            self.downsample = ConvBNReLU(use_attorch, in_dim, out_dim,
                                         stride=stride, relu=False)

    def forward(self, input: Tensor) -> Tensor:
        output = self.conv_bn_relu1(input)
        output = self.conv_bn_relu2(output)
        output = self.conv_bn_relu3(output, self.downsample(input))
        return output


def resnet_stage(
    use_attorch: bool,
    depth: int,
    in_dim: int,
    bottleneck_dim: int,
    stride: int = 1,
    expansion_factor: int = 4,
    ) -> nn.Sequential:
    """
    Creates a ResNet stage consisting of bottleneck blocks.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        depth: Depth.
        in_dim: Number of input channels to the first block.
        bottleneck_dim: Number of bottleneck channels.
        stride: Stride of the middle convolution of the first block.
        expansion_factor: Factor by which the bottleneck dimension is expanded
            to generate the output.
    """
    layer = nn.Sequential()
    for ind in range(depth):
        layer.append(BottleneckBlock(use_attorch, in_dim, bottleneck_dim,
                                     stride=stride if ind > 0 else 1,
                                     expansion_factor=expansion_factor))
        in_dim = expansion_factor * bottleneck_dim
    return layer


class ResNet(nn.Module):
    """
    Classifies the input using the ResNet architecture,
    optionally computing the loss if targets are passed.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        depths: Depths of the network's 4 stages.
        num_classes: Number of output classes.
    """
    def __init__(
        self,
        use_attorch: bool,
        depths: Tuple[int, int, int, int],
        num_classes: int = 1000,
        ) -> None:
        assert len(depths) == 4, \
            f'ResNet consists of 4 stages, received {len(depths)} depths instead'

        super().__init__()
        backend = attorch if use_attorch else nn

        self.stem = nn.Sequential(ConvBNReLU(use_attorch, 3, 64,
                                             kernel_size=7, stride=2),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = resnet_stage(use_attorch, depths[0], 64, 64)
        self.stage2 = resnet_stage(use_attorch, depths[1], 256, 128, stride=2)
        self.stage3 = resnet_stage(use_attorch, depths[1], 512, 256, stride=2)
        self.stage4 = resnet_stage(use_attorch, depths[1], 1024, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = backend.Linear(2048, num_classes)
        self.loss_func = backend.CrossEntropyLoss()

    def forward(
        self,
        input: Tensor,
        target: Optional[Tensor] = None,
        ) -> Tensor:
        output = self.stem(input)

        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)

        output = self.pool(output).flatten(1, -1)
        output = self.fc(output)

        return output if target is None else self.loss_func(output, target)


def resnet50(use_attorch: bool, num_classes: int = 1000) -> ResNet:
    """
    Returns a ResNet-50 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        num_classes: Number of output classes.
    """
    return ResNet(use_attorch, depths=(3, 4, 6, 3), num_classes=num_classes)


def resnet101(use_attorch: bool, num_classes: int = 1000) -> ResNet:
    """
    Returns a ResNet-101 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        num_classes: Number of output classes.
    """
    return ResNet(use_attorch, depths=(3, 4, 23, 3), num_classes=num_classes)


def resnet152(use_attorch: bool, num_classes: int = 1000) -> ResNet:
    """
    Returns a ResNet-152 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        num_classes: Number of output classes.
    """
    return ResNet(use_attorch, depths=(3, 8, 36, 3), num_classes=num_classes)

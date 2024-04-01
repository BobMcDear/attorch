"""
ConvNeXt for classification.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn

import attorch


class ConvNeXtStem(nn.Module):
    """
    Compresses the input using a ConvNeXt stem.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        out_dim: Number of output channels.
    """
    def __init__(self, use_attorch: bool, out_dim: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(3, out_dim, kernel_size=4, stride=4)
        self.norm = (attorch.LayerNorm(out_dim, eps=1e-6)
                     if use_attorch else nn.LayerNorm(out_dim, eps=1e-6))

    def forward(self, input: Tensor) -> Tensor:
        output = self.conv(input)
        output = output.permute(0, 2, 3, 1)
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)
        return output


class ConvNeXtDownsample(nn.Module):
    """
    Downsamples the input using a ConvNeXt downsampling layer.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        in_dim: Number of input channels.
        out_dim: Number of output channels.
    """
    def __init__(self, use_attorch: bool, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.norm = (attorch.LayerNorm(in_dim, eps=1e-6)
                     if use_attorch else nn.LayerNorm(in_dim, eps=1e-6))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, input: Tensor) -> Tensor:
        output = input.permute(0, 2, 3, 1)
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)
        output = self.conv(input)
        return output



class MLP(nn.Module):
    """
    Transforms the input using a multilayer perceptron with one hidden layer
    and the GELU activation function.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        in_dim: Number of input features.
        hidden_dim: Number of hidden features.
        out_dim: Number of output features.
            If None, it is set to the number of input features.
    """
    def __init__(
        self,
        use_attorch: bool,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        ) -> None:
        super().__init__()

        self.fc1 = (attorch.Linear(in_dim, hidden_dim, act_func='gelu')
                    if use_attorch else nn.Linear(in_dim, hidden_dim))
        self.act = nn.Identity() if use_attorch else nn.GELU()
        self.fc2 = (attorch.Linear(hidden_dim, out_dim or in_dim)
                    if use_attorch else nn.Linear(hidden_dim, out_dim or in_dim))

    def forward(self, input: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(input)))


class ConvNeXtBlock(nn.Module):
    """
    Passes the input through a ConvNeXt block.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        dim: Number of input and output channels.
        layer_scale_init_value: Initial value for LayerScale.
    """
    def __init__(
        self,
        use_attorch: bool,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        ) -> None:
        super().__init__()

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = (attorch.LayerNorm(dim, eps=1e-6)
                     if use_attorch else nn.LayerNorm(dim, eps=1e-6))
        self.mlp = MLP(use_attorch, dim, 4 * dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init_value))

    def forward(self, input: Tensor) -> Tensor:
        output = self.dw_conv(input)
        output = output.permute(0, 2, 3, 1)
        output = self.norm(output)
        output = self.mlp(output)
        output = self.gamma * output
        output = output.permute(0, 3, 1, 2)
        return input + output


def convnext_stage(
    use_attorch: bool,
    depth: int,
    in_dim: int,
    out_dim: Optional[int] = None,
    layer_scale_init_value: float = 1e-6,
    ) -> nn.Sequential:
    """
    Creates a ConvNeXt stage consisting of ConvNeXt blocks,
    optionally preceded by a downsampling layer.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        depth: Depth.
        in_dim: Number of input channels.
        out_dim: Number of output channels.
            If not None, a downsampling module with this output dimensionality
            is prepended to the stage.
        layer_scale_init_value: Initial value for LayerScale.
    """
    layer = nn.Sequential()
    if out_dim is not None:
        layer.append(ConvNeXtDownsample(use_attorch, in_dim, out_dim))
        in_dim = out_dim
    layer.extend([ConvNeXtBlock(use_attorch, in_dim, layer_scale_init_value)
                  for _ in range(depth)])
    return layer


class ConvNeXt(nn.Module):
    """
    Classifies the input using the ConvNeXt architecture,
    optionally computing the loss if targets are passed.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        depths: Depths of the network's 4 stages.
        dims: Number of channels per stage.
        layer_scale_init_value: Initial value for LayerScale.
        num_classes: Number of output classes.
    """
    def __init__(
        self,
        use_attorch: bool,
        depths: Tuple[int, ...],
        dims: Tuple[int, ...],
        layer_scale_init_value: float = 1e-6,
        num_classes: int = 1000,
        ) -> None:
        assert len(depths) == 4, \
            f'ConvNeXt consists of 4 stages, received {len(depths)} depths instead'
        assert len(dims) == 4, \
            f'ConvNeXt consists of 4 stages, received {len(dims)} widths instead'

        super().__init__()
        backend = attorch if use_attorch else nn

        self.stem = ConvNeXtStem(use_attorch, dims[0])
        self.stage1 = convnext_stage(use_attorch, depths[0], dims[0],
                                     layer_scale_init_value=layer_scale_init_value)
        self.stage2 = convnext_stage(use_attorch, depths[1], dims[0], dims[1],
                                     layer_scale_init_value=layer_scale_init_value)
        self.stage3 = convnext_stage(use_attorch, depths[2], dims[1], dims[2],
                                     layer_scale_init_value=layer_scale_init_value)
        self.stage4 = convnext_stage(use_attorch, depths[3], dims[2], dims[3],
                                     layer_scale_init_value=layer_scale_init_value)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = backend.LayerNorm(dims[3], eps=1e-6)
        self.fc = backend.Linear(dims[3], num_classes)
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
        output = self.norm(output)
        output = self.fc(output)

        return output if target is None else self.loss_func(output, target)


def convnext_tiny(use_attorch: bool, num_classes: int = 1000) -> ConvNeXt:
    """
    Returns a ConvNeXt-Tiny classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        num_classes: Number of output classes.
    """
    return ConvNeXt(use_attorch, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                    num_classes=num_classes)


def convnext_small(use_attorch: bool, num_classes: int = 1000) -> ConvNeXt:
    """
    Returns a ConvNeXt-Small classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        num_classes: Number of output classes.
    """
    return ConvNeXt(use_attorch, depths=(3, 3, 27, 3), dims=(96, 192, 384, 768),
                    num_classes=num_classes)


def convnext_base(use_attorch: bool, num_classes: int = 1000) -> ConvNeXt:
    """
    Returns a ConvNeXt-Base classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        num_classes: Number of output classes.
    """
    return ConvNeXt(use_attorch, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024),
                    num_classes=num_classes)


def convnext_large(use_attorch: bool, num_classes: int = 1000) -> ConvNeXt:
    """
    Returns a ConvNeXt-Large classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        num_classes: Number of output classes.
    """
    return ConvNeXt(use_attorch, depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536),
                    num_classes=num_classes)


def convnext_xlarge(use_attorch: bool, num_classes: int = 1000) -> ConvNeXt:
    """
    Returns a ConvNeXt-XLarge classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        num_classes: Number of output classes.
    """
    return ConvNeXt(use_attorch, depths=(3, 3, 27, 3), dims=(256, 512, 1024, 2048),
                    num_classes=num_classes)

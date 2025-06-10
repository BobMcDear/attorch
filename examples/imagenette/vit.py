"""
ViT for classification.
"""


from typing import Optional

import torch
from torch import Tensor
from torch import nn

import attorch


class PatchEmbed(nn.Module):
    """
    Projects patches of images onto an embedding space.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        dim: Embedding dimension.
        patch_size: Size of each patch along both spatial dimensions.
    """
    def __init__(
        self,
        use_attorch: bool,
        dim: int,
        patch_size: int = 16,
        ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.proj = (attorch.Linear(3 * patch_size ** 2, dim)
                     if use_attorch else nn.Linear(3 * patch_size ** 2, dim))

    def forward(self, input: Tensor) -> Tensor:
        batch_dim, _, height, width = input.shape
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size

        input = input.view(batch_dim, 3,
                           num_patches_height, self.patch_size,
                           num_patches_width, self.patch_size)
        input = input.permute(0, 2, 4, 3, 5, 1).contiguous()
        input = input.view(batch_dim, num_patches_height * num_patches_width, -1)

        return self.proj(input)


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


class TransformerBlock(nn.Module):
    """
    Passes the input through a transformer block.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        dim: Embedding dimension.
        num_heads: Number of heads for multi-headed self-attention.
    """
    def __init__(
        self,
        use_attorch: bool,
        dim: int,
        num_heads: int,
        ) -> None:
        super().__init__()
        self.use_attorch = use_attorch
        backend = attorch if use_attorch else nn

        self.ln1 = backend.LayerNorm(dim)
        self.attn = backend.MultiheadAttention(dim, num_heads,
                                               batch_first=True)

        self.ln2 = backend.LayerNorm(dim)
        self.mlp = MLP(use_attorch, dim, 4 * dim)

    def forward(self, input: Tensor) -> Tensor:
        if self.use_attorch:
            output = input + self.attn(self.ln1(input))

        else:
            output = self.ln1(input)
            output = input + self.attn(output, output, output,
                                       need_weights=False)[0]

        output = output + self.mlp(self.ln2(output))
        return output


class ViT(nn.Module):
    """
    Classifies the input using the ViT architecture,
    optionally computing the loss if targets are passed.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        depth: Depth of the transformer.
        dim: Embedding dimension.
        num_heads: Number of heads for multi-headed self-attention.
        image_size: Height and width of input images.
        patch_size: Size of each patch along both spatial dimensions.
        num_classes: Number of output classes.
    """
    def __init__(
        self,
        use_attorch: bool,
        depth: int,
        dim: int,
        num_heads: int,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        ) -> None:
        super().__init__()
        backend = attorch if use_attorch else nn

        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Embedding(num_patches, dim)
        self.patch_embed = PatchEmbed(use_attorch, dim, patch_size)
        self.transformer = nn.Sequential(*[TransformerBlock(use_attorch, dim, num_heads)
                                           for _ in range(depth)])
        self.norm = backend.LayerNorm(dim)
        self.fc = backend.Linear(dim, num_classes)
        self.loss_func = backend.CrossEntropyLoss()

    def forward(
        self,
        input: Tensor,
        target: Optional[Tensor] = None,
        ) -> Tensor:
        patch_embed = self.patch_embed(input)
        pos_embed = self.pos_embed(torch.arange(0, patch_embed.shape[1],
                                                dtype=torch.long,
                                                device=input.device))

        output = self.transformer(patch_embed + pos_embed)
        output = torch.mean(output, dim=1)
        output = self.norm(output)
        output = self.fc(output)

        return output if target is None else self.loss_func(output, target)


def vit_tiny_patch16(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Tiny-Patch16 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=192, num_heads=3,
               image_size=image_size, patch_size=16,
               num_classes=num_classes)


def vit_xsmall_patch16(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-XSmall-Patch16 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=10, dim=256, num_heads=4,
               image_size=image_size, patch_size=16,
               num_classes=num_classes)


def vit_small_patch32(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Small-Patch32 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=384, num_heads=6,
               image_size=image_size, patch_size=32,
               num_classes=num_classes)


def vit_small_patch16(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Small-Patch16 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=384, num_heads=6,
               image_size=image_size, patch_size=16,
               num_classes=num_classes)


def vit_small_patch8(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Small-Patch8 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=384, num_heads=6,
               image_size=image_size, patch_size=8,
               num_classes=num_classes)


def vit_medium_patch32(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Medium-Patch32 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=512, num_heads=8,
               image_size=image_size, patch_size=32,
               num_classes=num_classes)


def vit_medium_patch16(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Medium-Patch16 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=512, num_heads=8,
               image_size=image_size, patch_size=16,
               num_classes=num_classes)


def vit_base_patch32(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Base-Patch32 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=768, num_heads=12,
               image_size=image_size, patch_size=32,
               num_classes=num_classes)


def vit_base_patch16(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Base-Patch16 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=768, num_heads=12,
               image_size=image_size, patch_size=16,
               num_classes=num_classes)


def vit_base_patch8(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Base-Patch8 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=12, dim=768, num_heads=12,
               image_size=image_size, patch_size=8,
               num_classes=num_classes)


def vit_large_patch32(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Large-Patch32 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=24, dim=1024, num_heads=16,
               image_size=image_size, patch_size=32,
               num_classes=num_classes)


def vit_large_patch16(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Large-Patch16 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=24, dim=1024, num_heads=16,
               image_size=image_size, patch_size=16,
               num_classes=num_classes)


def vit_large_patch14(
    use_attorch: bool,
    image_size: int = 224,
    num_classes: int = 1000,
    ) -> ViT:
    """
    Returns a ViT-Large-Patch14 classifier with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        image_size: Height and width of input images.
        num_classes: Number of output classes.
    """
    return ViT(use_attorch, depth=24, dim=1024, num_heads=16,
               image_size=image_size, patch_size=14,
               num_classes=num_classes)

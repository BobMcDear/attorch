"""
Multilayer perceptron (MLP) for MNIST classification.
"""


from typing import Optional

from torch import Tensor
from torch import nn

import attorch


class MLP(nn.Module):
    """
    Transforms the input using a multilayer perceptron (MLP) with an arbitrary
    number of hidden layers, optionally computing the loss if targets are passed.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        in_dim: Number of input features.
        hidden_dim: Number of hidden features.
        depth: Number of hidden layers.
        num_classes: Number of output classes.
    """
    def __init__(
        self,
        use_attorch: bool,
        in_dim: int = 784,
        hidden_dim: int = 128,
        depth: int = 1,
        num_classes: int = 10,
        ) -> None:
        super().__init__()

        backend = attorch if use_attorch else nn
        layer_fn = lambda dim: ([attorch.Linear(dim, hidden_dim, act_func='relu')]
                                if use_attorch else [nn.Linear(dim, hidden_dim), nn.ReLU()])

        layers = layer_fn(in_dim)
        for _ in range(depth - 1):
            layers += layer_fn(hidden_dim)
        layers.append(backend.Linear(hidden_dim, num_classes))

        self.layers = nn.Sequential(*layers)
        self.loss_func = backend.CrossEntropyLoss()

    def forward(self, input: Tensor, target: Optional[Tensor] = None) -> Tensor:
        output = self.layers(input.flatten(start_dim=1))
        return output if target is None else self.loss_func(output, target)

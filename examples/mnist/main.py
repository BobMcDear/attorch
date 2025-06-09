"""
Trains and benchmarks a multilayer perceptron (MLP) on the MNIST dataset.
"""


import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST

from .mlp import MLP
from ..utils import AvgMeter, benchmark_fw_and_bw


def create_dls(batch_size: int = 1_024) -> Tuple[DataLoader, DataLoader]:
    """
    Creates data loaders for MNIST with normalization.

    args:
        batch_size: Batch size.

    Returns:
        Training and validation MNIST data loaders.
    """
    transform = T.Compose([T.ToTensor(),
                           T.Normalize((0.1307,), (0.3081,))])

    train_dataset = MNIST(root='.', train=True, transform=transform,
                          download=not Path('MNIST/').exists())
    valid_dataset = MNIST(root='.', train=False, transform=transform)

    train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                          shuffle=False, drop_last=True)

    return train_dl, valid_dl


def train(
    model: nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    epochs: int = 10,
    batch_size: int = 1_024,
    ) -> float:
    """
    Trains and validates a model for classification.

    Args:
        model: Model to train. Its forward pass must optionally accept targets
            to compute the loss.
        train_dl: Data loader for training.
        valid_dl: Data loader for validation.
        epochs: Number of epochs to train for.
        batch_size: Batch size.

    Returns:
        Total training and validation time.
    """
    model = model.to('cuda')
    optim = SGD(model.parameters(), lr=batch_size / 64 * 0.005)
    optim.zero_grad()

    avg_meter = AvgMeter()
    start = time.time()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        model.train()
        avg_meter.reset()
        for input, target in train_dl:
            input = input.to('cuda')
            target = target.to('cuda')

            loss = model(input, target)
            optim.zero_grad()

            avg_meter.update(loss.item(), len(input))
        print(f'Training loss: {avg_meter.avg}')

        model.eval()
        avg_meter.reset()
        with torch.no_grad():
            for input, target in valid_dl:
                input = input.to('cuda')
                target = target.to('cuda')

                output = model(input)
                acc = (output.argmax(dim=-1) == target).float().mean()
                avg_meter.update(acc.item(), len(input))
        print(f'Validation accuracy: {avg_meter.avg}')

    return time.time() - start


def main(model_cls: Callable, epochs: int = 10, batch_size: int = 1_024) -> None:
    """
    Trains and benchmarks a vision model on the MNIST dataset.

    Args:
        model_cls: Model class to train, with a 'num_classes' argument for
            specifying the number of output classes.
        epochs: Number of epochs to train for.
        batch_size: Batch size.
    """
    train_dl, valid_dl = create_dls(batch_size)
    model = model_cls(num_classes=len(MNIST.classes)).to('cuda')

    input, target = next(iter(train_dl))
    input = input.to('cuda')
    target = target.to('cuda')

    for _ in range(10):
        model.train()
        with torch.autocast('cuda'):
            loss = model(input, target)
        loss.backward()

        model.eval()
        with torch.no_grad() and torch.autocast('cuda'):
            model(input)

    model.train()
    benchmark_fw_and_bw(model, input=input, target=target)

    print('Total training and validation time: '
          f'{train(model, train_dl, valid_dl, epochs, batch_size)}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Trains and benchmarks a multilayer perceptron (MLP) on the MNIST dataset.')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=128,
                        help='Number of hidden features in the MLP.')
    parser.add_argument('--depth',
                        type=int,
                        default=1,
                        help='Number of hidden layers in the MLP.')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1_024,
                        help='Batch size')
    args = parser.parse_args()

    model_cls = partial(MLP, hidden_dim=args.hidden_dim, depth=args.depth)

    print('attorch run:')
    main(partial(model_cls, use_attorch=True),
         epochs=args.epochs, batch_size=args.batch_size)

    print('PyTorch run:')
    main(partial(model_cls, use_attorch=False),
         epochs=args.epochs, batch_size=args.batch_size)

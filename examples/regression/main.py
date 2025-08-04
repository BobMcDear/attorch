"""
Trains and benchmarks a multilayer perceptron (MLP) on synthetic regression data.
"""


import time
from argparse import ArgumentParser
from types import ModuleType
from typing import Callable, Tuple

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

import attorch
from ..utils import AvgMeter, benchmark_fw_and_bw


def create_dls(
    n_samples: int = 50_000,
    dim: int = 10,
    batch_size: int = 1_024,
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Creates synthetic regression data loaders.

    Args:
        n_samples: Number of samples to generate.
        dim: Dimensionality of synthetic data.
        batch_size: Batch size.

    Returns:
        Training and validation data loaders.
    """
    torch.manual_seed(0)
    input = torch.randn(n_samples, dim)
    target = input @ torch.randn(dim, 1) + 0.1 * torch.randn(n_samples, 1)
    n_train = int(0.8 * n_samples)

    train_dataset = TensorDataset(input[:n_train], target[:n_train])
    valid_dataset = TensorDataset(input[n_train:], target[n_train:])

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          drop_last=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          drop_last=True)

    return train_dl, valid_dl


def train(
    model: torch.nn.Module,
    loss_fn: Callable,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    epochs: int = 10,
    batch_size: int = 1_024,
    ) -> float:
    """
    Trains and validates a regression model.

    Args:
        model: Model to train.
        loss_fn: Loss function.
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

            loss = loss_fn(model(input), target)
            optim.zero_grad()

            avg_meter.update(loss.item(), len(input))
        print(f'Training loss: {avg_meter.avg}')

        model.eval()
        avg_meter.reset()
        with torch.no_grad():
            for input, target in valid_dl:
                input = input.to('cuda')
                target = target.to('cuda')

                loss = loss_fn(model(input), target)
                avg_meter.update(loss.item(), len(input))
        print(f'Validation loss: {avg_meter.avg}')

    return time.time() - start


def main(
    nn: ModuleType,
    n_samples: int = 50_000,
    dim: int = 10,
    epochs: int = 10,
    batch_size: int = 1_024,
    ) -> None:
    """
    Trains and benchmarks a regression MLP on synthetic data.

    Args:
        nn: Neural network module used to construct the MLP.
        n_samples: Number of samples to generate.
        dim: Dimensionality of synthetic data.
        epochs: Number of epochs to train for.
        batch_size: Batch size.
    """
    train_dl, valid_dl = create_dls(n_samples, dim, batch_size)
    model = nn.Sequential(nn.Linear(dim, dim // 2),
                          nn.ReLU(),
                          nn.Linear(dim // 2, 1)).to('cuda')
    loss_fn = nn.MSELoss()

    input, target = next(iter(train_dl))
    input = input.to('cuda')
    target = target.to('cuda')

    for _ in range(10):
        model.train()
        with torch.autocast('cuda'):
            loss = loss_fn(model(input), target)
        loss.backward()

        model.eval()
        with torch.no_grad() and torch.autocast('cuda'):
            model(input)

    model.train()
    benchmark_fw_and_bw(model, input=input)

    print('Total training and validation time: '
          f'{train(model, loss_fn, train_dl, valid_dl, epochs, batch_size)}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Trains and benchmarks a multilayer perceptron (MLP) on synthetic regression data.')
    parser.add_argument('--n_samples',
                        type=int,
                        default=50_000,
                        help='Number of samples to generate')
    parser.add_argument('--dim',
                        type=int,
                        default=10,
                        help='Dimensionality of synthetic data')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1_024,
                        help='Batch size')
    args = parser.parse_args()

    print('attorch run:')
    main(attorch.nn, n_samples=args.n_samples, dim=args.dim,
         epochs=args.epochs, batch_size=args.batch_size)

    print('PyTorch run:')
    main(torch.nn, n_samples=args.n_samples, dim=args.dim,
         epochs=args.epochs, batch_size=args.batch_size)

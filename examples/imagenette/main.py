"""
Trains and benchmarks a vision model on the Imagenette classification dataset.
"""


import time
from argparse import ArgumentParser
from functools import partial
from math import sqrt
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import Imagenette

from .convnext import convnext_atto, convnext_femto, convnext_pico, convnext_nano, \
    convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from .resnet import resnet14, resnet26, resnet50, resnet101, resnet152
from .vit import vit_tiny_patch16, vit_xsmall_patch16, vit_small_patch32, \
    vit_small_patch16, vit_small_patch8, vit_medium_patch32, vit_medium_patch16, \
    vit_base_patch32, vit_base_patch16, vit_base_patch8, \
    vit_large_patch32, vit_large_patch16, vit_large_patch14
from ..utils import AvgMeter, benchmark_fw_and_bw


def create_dls(
    batch_size: int = 32,
    center_crop_size: int = 256,
    image_size: int = 224,
    num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Creates data loaders for Imagenette with appropriate transforms.

    args:
        batch_size: Batch size.
        center_crop_size: Center crop size for validation.
        image_size: Input image size.
        num_workers: Number of workers for data loading.

    Returns:
        Training and validation Imagenette data loaders.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([T.RandomResizedCrop(image_size),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor(),
                                 T.Normalize(mean=mean, std=std)])
    valid_transform = T.Compose([T.CenterCrop(center_crop_size),
                                 T.Resize(image_size),
                                 T.ToTensor(),
                                 T.Normalize(mean=mean, std=std)])

    train_dataset = Imagenette(root='.', split='train', download=not Path('imagenette2/').exists(),
                               transform=train_transform)
    valid_dataset = Imagenette(root='.', split='val', download=False,
                               transform=valid_transform)

    train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, pin_memory=True,
                          drop_last=True)
    valid_dl = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True,
                          drop_last=True)

    return train_dl, valid_dl


def train(
    model: nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    epochs: int = 10,
    batch_size: int = 32,
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
    optim = AdamW(model.parameters(), lr=1e-4)
    optim.zero_grad()
    sched = OneCycleLR(optim, max_lr=sqrt(batch_size / 32) * 4e-4,
                       steps_per_epoch=len(train_dl), epochs=epochs)
    scaler = torch.GradScaler('cuda')

    avg_meter = AvgMeter()
    start = time.time()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        model.train()
        avg_meter.reset()
        for ind, (input, target) in enumerate(train_dl):
            if (ind + 1) % 5 == 0:
                print(f'Training iteration {ind+1}/{len(train_dl)}', end='\r')

            input = input.to('cuda')
            target = target.to('cuda')

            with torch.autocast('cuda'):
                loss = model(input, target)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            optim.zero_grad()

            avg_meter.update(loss.item(), len(input))
        print(f'Training loss: {avg_meter.avg}')

        model.eval()
        avg_meter.reset()
        with torch.no_grad():
            for ind, (input, target) in enumerate(valid_dl):
                if (ind + 1) % 5 == 0:
                    print(f'Validation iteration {ind+1}/{len(valid_dl)}', end='\r')

                input = input.to('cuda')
                target = target.to('cuda')

                with torch.autocast('cuda'):
                    output = model(input)
                acc = (output.argmax(dim=-1) == target).float().mean()
                avg_meter.update(acc.item(), len(input))
        print(f'Validation accuracy: {avg_meter.avg}')

    return time.time() - start


def main(
    model_cls: Callable,
    epochs: int = 10,
    batch_size: int = 32,
    center_crop_size: int = 256,
    image_size: int = 224,
    num_workers: int = 4,
    ) -> None:
    """
    Trains and benchmarks a vision model on the Imagenette classification dataset.

    Args:
        model_cls: Model class to train, with a 'num_classes' argument for
            specifying the number of output classes.
        epochs: Number of epochs to train for.
        batch_size: Batch size.
        center_crop_size: Center crop size for validation.
        image_size: Input image size.
        num_workers: Number of workers for data loading.
    """
    train_dl, valid_dl = create_dls(batch_size, center_crop_size, image_size, num_workers)
    model = model_cls(num_classes=len(Imagenette._WNID_TO_CLASS)).to('cuda')

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
    parser = ArgumentParser(description='Trains and benchmarks a vision model on the Imagenette classification dataset.')
    parser.add_argument('--model',
                        type=str,
                        default='resnet101',
                        choices=['resnet14', 'resnet26', 'resnet50', 'resnet101', 'resnet152',
                                 'convnext_atto', 'convnext_femto',
                                 'convnext_pico', 'convnext_nano',
                                 'convnext_tiny', 'convnext_small',
                                 'convnext_base', 'convnext_large',
                                 'convnext_xlarge',
                                 'vit_tiny_patch16', 'vit_xsmall_patch16',
                                 'vit_small_patch32', 'vit_small_patch16', 'vit_small_patch8',
                                 'vit_medium_patch32', 'vit_medium_patch16',
                                 'vit_base_patch32', 'vit_base_patch16', 'vit_base_patch8',
                                 'vit_large_patch32', 'vit_large_patch16', 'vit_large_patch14'],
                        help='Name of vision model to train')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')
    parser.add_argument('--center_crop_size',
                        type=int,
                        default=256,
                        help='Center crop size for validation')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Input image size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of workers for data loading')
    args = parser.parse_args()

    model_cls = (partial(locals()[args.model], image_size=args.image_size)
                 if 'vit' in args.model else partial(locals()[args.model]))

    print('attorch run:')
    main(partial(model_cls, use_attorch=True),
         epochs=args.epochs, batch_size=args.batch_size,
         center_crop_size=args.center_crop_size, image_size=args.image_size,
         num_workers=args.num_workers)

    print('PyTorch run:')
    main(partial(model_cls, use_attorch=False),
         epochs=args.epochs, batch_size=args.batch_size,
         center_crop_size=args.center_crop_size, image_size=args.image_size,
         num_workers=args.num_workers)

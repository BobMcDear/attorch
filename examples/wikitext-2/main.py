"""
Trains and benchmarks a language model on the WikiText-2 dataset.
"""


import time
from argparse import ArgumentParser
from functools import partial
from math import exp, sqrt
from typing import Callable, Tuple

import torch
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .gpt import gpt2
from ..utils import AvgMeter, benchmark_fw_and_bw


def create_dls(
    batch_size: int = 32,
    seq_len: int = 512,
    num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Creates data loaders for WikiText-2 with tokenization.

    args:
        batch_size: Batch size.
        seq_len: Sequence length.
        num_workers: Number of workers for data loading.

    Returns:
        Training and validation WikiText-2 data loaders.
    """
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    dataset = dataset.map(lambda input: tokenizer(input['text'],
                                                  max_length=seq_len,
                                                  padding='max_length',
                                                  truncation=True),
                          batched=True)
    dataset.set_format('torch')

    train_dl = DataLoader(dataset=dataset['train'], batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          drop_last=True)
    valid_dl = DataLoader(dataset=dataset['validation'], batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
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
    Trains and validates a model for language modelling.

    Args:
        model: Model to train. Its forward pass must optionally accept return_loss
            to compute the loss.
        train_dl: Data loader for training.
        valid_dl: Data loader for validation.
        epochs: Number of epochs to train for.
        batch_size: Batch size.

    Returns:
        Total training and validation time.
    """
    model = model.to('cuda')
    optim = AdamW(model.parameters(), lr=sqrt(batch_size / 32) * 4e-4)
    optim.zero_grad()
    scaler = torch.GradScaler('cuda')

    avg_meter = AvgMeter()
    start = time.time()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        model.train()
        avg_meter.reset()
        for batch in train_dl:
            input = batch['input_ids'].to('cuda')

            with torch.autocast('cuda'):
                loss = model(input, return_loss=True)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            avg_meter.update(loss.item(), len(input))
        print(f'Training loss: {avg_meter.avg}')

        model.eval()
        avg_meter.reset()
        with torch.no_grad():
            for batch in valid_dl:
                input = batch['input_ids'].to('cuda')

                with torch.autocast('cuda'):
                    loss = model(input, return_loss=True)
                avg_meter.update(loss.item(), len(input))
        print(f'Validation perplexity: {exp(avg_meter.avg)}')

    return time.time() - start


def main(
    model_cls: Callable,
    epochs: int = 10,
    batch_size: int = 32,
    seq_len: int = 512,
    num_workers: int = 4,
    ) -> None:
    """
    Trains and benchmarks a language model on the WikiText-2 dataset.

    Args:
        model_cls: Model class to train, with a 'vocab_size' argument for
            specifying the vocabulary size and a 'max_seq_len' argument for
            specifying the maximum sequence length of the incoming inputs.
        epochs: Number of epochs to train for.
        batch_size: Batch size.
        seq_len: Sequence length.
        num_workers: Number of workers for data loading.
    """
    train_dl, valid_dl = create_dls(batch_size, seq_len, num_workers)
    vocab_size = AutoTokenizer.from_pretrained('gpt2').vocab_size
    model = model_cls(vocab_size=vocab_size, max_seq_len=seq_len).to('cuda')

    batch = next(iter(train_dl))
    input = batch['input_ids'].to('cuda')

    for _ in range(10):
        model.train()
        with torch.autocast('cuda'):
            loss = model(input, return_loss=True)
        loss.backward()

        model.eval()
        with torch.no_grad() and torch.autocast('cuda'):
            model(input, return_loss=True)

    model.train()
    benchmark_fw_and_bw(model, input=input, return_loss=True)

    print('Total training and validation time: '
          f'{train(model, train_dl, valid_dl, epochs, batch_size)}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Trains and benchmarks a language model on the WikiText-2 dataset.')
    parser.add_argument('--model',
                        type=str,
                        default='gpt2',
                        choices=['gpt2'],
                        help='Name of language model to train')
    parser.add_argument('--downsize',
                        type=int,
                        default=1,
                        help='The depth and width of the model are calculated by dividing GPT2\'s original depth and width by this factor.')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')
    parser.add_argument('--seq_len',
                        type=int,
                        default=512,
                        help='Sequence length')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of workers for data loading')
    args = parser.parse_args()

    print('attorch run:')
    main(partial(locals()[args.model], use_attorch=True, downsize=args.downsize),
         epochs=args.epochs, batch_size=args.batch_size,
         seq_len=args.seq_len, num_workers=args.num_workers)

    print('PyTorch run:')
    main(partial(locals()[args.model], use_attorch=False, downsize=args.downsize),
         epochs=args.epochs, batch_size=args.batch_size,
         seq_len=args.seq_len, num_workers=args.num_workers)

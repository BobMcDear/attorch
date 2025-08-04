# Synthetic Regression

This example trains and benchmarks a multilayer perceptron (MLP) on synthetic regression data. It relies on ```attorch.nn``` to demonstrate how attorch can be used as a drop-in replacement for PyTorch without kernel fusion or other enhancements.

## Requirements

This example has no requirements aside from attorch and its dependencies.

## Training

To run this example, please run ```python -m examples.regression.main``` from the root directory. The arguments are as follows.
* ```--n_samples```: Number of samples to generate.
* ```--dim```: Dimensionality of synthetic data.
* ```--epochs```: Number of epochs to train for.
* ```--batch_size```: Batch size.

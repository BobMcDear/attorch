# MNIST Classification

This example trains and benchmarks a multilayer perceptron (MLP) on the MNIST dataset.

## Requirements

The requirement for this example, aside from attorch and its dependencies, is,

* ```torchvision==0.19.0```

## Training

To run this example, please run ```python -m examples.mnist.main``` from the root directory. The arguments are as follows.
* ```--hidden_dim```: Number of hidden features in the MLP.
* ```--depth```: Number of hidden layers in the MLP.
* ```--epochs```: Number of epochs to train for.
* ```--batch_size```: Batch size.

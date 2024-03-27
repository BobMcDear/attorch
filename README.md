# attorch

• **[Introduction](#introduction)**<br>
• **[Installation](#installation)**<br>
• **[Layers](#layers)**<br>
• **[PyTorch Fallback](#pytorch-fallback)**<br>
• **[Tests](#tests)**<br>

## Introduction

attorch is a subset of PyTorch's [```nn```](https://pytorch.org/docs/stable/nn.html) module, written purely in Python using OpenAI's [Triton](https://github.com/openai/triton). Its goal is to be an easily hackable, self-contained, and readable collection of neural network modules whilst maintaining or improving upon the efficiency of PyTorch. In other words, it intends to be a forkable project endowed with a simple, intuitive design that can serve as an accessible starting point for those who are seeking to develop custom deep learning operations but are not satisfied with the speed of a pure PyTorch implementation and do not have the technical expertise or resources to write CUDA kernels.

There already exist a number of wonderful PyTorch-like frameworks powered by Triton, but most concentrate solely on Transformers and NLP applications, whereas attorch aims to be more inclusive by also presenting a variety of layers pertaining to areas besides NLP such as computer vision. Moreover, attorch is not an inference-only package and fully supports both forward and backward passes, meaning it can be used during training as well as inference, though its performance for the latter is generally not on par with dedicated inference engines.

## Installation

The only dependencies of attorch are ```torch==2.2.0``` and ```triton==2.2.0```. Please install the specified versions of these two libraries and clone this repository to get started.

## Layers

Currently implemented layers, with automatic mixed precision (AMP) support, are,

* ```attorch.Conv2d```: 2D-convolves over the input using weights, optionally adding bias.
* ```attorch.GELU```: Applies GELU to the input.
* ```attorch.ReLU```: Applies ReLU to the input.
* ```attorch.SiLU```: Applies SiLU to the input.
* ```attorch.Sigmoid```: Applies sigmoid to the input.
* ```attorch.Tanh```: Applies tanh to the input.
* ```attorch.LogSoftmax```: Normalizes the input using softmax and takes its log.
* ```attorch.Softmax```: Normalizes the input using softmax.
* ```attorch.Softmin```: Normalizes the input using softmin.
* ```attorch.BatchNorm1d```: Batch-normalizes the 2D or 3D input, optionally fusing an activation function and adding a residual to the pre-activation result.
* ```attorch.BatchNorm2d```: Batch-normalizes the 4D input, optionally fusing an activation function and adding a residual to the pre-activation result.
* ```attorch.LayerNorm```: Layer-normalizes the input.
* ```attorch.Linear```: Linearly transforms the input using weights, optionally adding bias and fusing an activation function.
* ```attorch.Dropout```: Randomly zeroes elements in the input during training.
* ```attorch.L1Loss```: Measures the mean absolute error between the input and target.
* ```attorch.MSELoss```: Measures the mean squared error between the input and target.
* ```attorch.CrossEntropyLoss```: Measures the mean cross entropy loss between the input and target, with optional reweighing of each class.
* ```attorch.NLLLoss```: Measures the negative log likelihood loss between the input and target, with optional reweighing of each class.

Unless otherwise noted in their docstrings, the aforementioned layers behave identically to their PyTorch equivalents.

## PyTorch Fallback

To enable easier integration of attorch and PyTorch layers, ```attorch.nn``` is offered, which provides an interface to attorch's modules with PyTorch fallback should a desired layer not be available, as seen below.

```python
from attorch import nn


lin = nn.Linear(10, 20) # Uses attorch's linear layer
gap = nn.AdaptiveAvgPool2d(1) # Uses PyTorch's global pooling since GAP is not available in attorch
```

## Tests

Each module can be tested against its PyTorch counterpart to ensure correctness. These tests are included under ```tests/``` and can be executed using ```pytest```. It should be noted that some might fail owing to numerical precision issues, but in most practical use cases, that should not be a problem.

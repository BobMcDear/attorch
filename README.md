# attorch

• **[Introduction](#introduction)**<br>
• **[Installation](#installation)**<br>
• **[Layers](#layers)**<br>
• **[Math Functions](#math-functions)**<br>
• **[PyTorch Fallback](#pytorch-fallback)**<br>
• **[Tests](#tests)**<br>

## Introduction

attorch is a subset of PyTorch's [```nn```](https://pytorch.org/docs/stable/nn.html) module, written purely in Python using OpenAI's [Triton](https://github.com/openai/triton). Its goal is to be an easily hackable, self-contained, and readable collection of neural network modules whilst maintaining or improving upon the efficiency of PyTorch. In other words, it intends to be a forkable project endowed with a simple, intuitive design that can serve as an accessible starting point for those who are seeking to develop custom deep learning operations but are not satisfied with the speed of a pure PyTorch implementation and do not have the technical expertise or resources to write CUDA kernels.

There already exist a number of wonderful PyTorch-like frameworks powered by Triton, including [kernl](https://github.com/ELS-RD/kernl/tree/main), [xFormers](https://github.com/facebookresearch/xformers), [Unsloth](https://github.com/unslothai/unsloth), and [```fla```](https://github.com/sustcsonglin/flash-linear-attention), but most concentrate mainly on Transformers and NLP applications, whereas attorch aims to be more inclusive by also presenting a variety of layers pertaining to areas besides NLP such as computer vision. Moreover, attorch is not an inference-only package and fully supports both forward and backward passes, meaning it can be used during training as well as inference, though its performance for the latter is generally not on par with dedicated inference engines.

## Installation

The only dependencies of attorch are ```torch==2.4.0``` and ```triton==3.0.0```. Please install the specified versions of these two libraries and clone this repository to get started.

## Layers

Currently implemented layers, with automatic mixed precision (AMP) support, are,

* ```attorch.Conv1d```: 1D-convolves over the input using weights, optionally adding bias.
* ```attorch.Conv2d```: 2D-convolves over the input using weights, optionally adding bias.
* ```attorch.MultiheadAttention```: Applies multi-headed scaled dot-product attention to the inputs.
* ```attorch.Hardsigmoid```: Applies hard sigmoid to the input, optionally fusing dropout.
* ```attorch.Hardswish```: Applies hard Swish to the input, optionally fusing dropout.
* ```attorch.GELU```: Applies GELU to the input, optionally fusing dropout.
* ```attorch.ReLU```: Applies ReLU to the input, optionally fusing dropout.
* ```attorch.ReLU6```: Applies ReLU6 to the input, optionally fusing dropout.
* ```attorch.SELU```: Applies SELU to the input, optionally fusing dropout.
* ```attorch.SiLU```: Applies SiLU to the input, optionally fusing dropout.
* ```attorch.Mish```: Applies Mish to the input, optionally fusing dropout.
* ```attorch.Sigmoid```: Applies sigmoid to the input, optionally fusing dropout.
* ```attorch.Tanh```: Applies tanh to the input, optionally fusing dropout.
* ```attorch.GLU```: Applies the gated linear unit with an arbitrary activation function to the input.
* ```attorch.LogSoftmax```: Normalizes the input using softmax and takes its log.
* ```attorch.Softmax```: Normalizes the input using softmax.
* ```attorch.Softmin```: Normalizes the input using softmin.
* ```attorch.BatchNorm1d```: Batch-normalizes the 2D or 3D input, optionally fusing an activation function and adding a residual to the pre-activation result.
* ```attorch.BatchNorm2d```: Batch-normalizes the 4D input, optionally fusing an activation function and adding a residual to the pre-activation result.
* ```attorch.LayerNorm```: Layer-normalizes the input.
* ```attorch.RMSNorm```: Root-mean-square-normalizes the input.
* ```attorch.Linear```: Linearly transforms the input using weights, optionally adding bias and fusing an activation function.
* ```attorch.Dropout```: Randomly zeroes elements in the input during training.
* ```attorch.L1Loss```: Measures the mean absolute error between the input and target.
* ```attorch.MSELoss```: Measures the mean squared error between the input and target.
* ```attorch.CrossEntropyLoss```: Measures the mean cross entropy loss between the input and target, with optional reweighing of each class.
* ```attorch.NLLLoss```: Measures the negative log likelihood loss between the input and target, with optional reweighing of each class.

Unless otherwise noted in their docstrings, the aforementioned layers behave identically to their PyTorch equivalents.

## Math Functions
Triton kernels are generally composed of two parts: One handles the loading and storing of the relevant tensors, the other transforms the data using appropriate mathematical functions. For instance, a layer normalization kernel reads one or several rows from the input (load), standardizes the features (math), and writes the results into a container (store). A selection of these pure math functions is supplied by ```attorch.math```, the objective being to faciliate the implementation of custom kernels and operation fusion. Although only the forward passes of the said functions are available in ```attorch.math```, thanks to their purity and absence of I/O actions, their gradients can be automatically derived via the [```triton-autodiff```](https://github.com/srush/triton-autodiff) library. Significant portions of attorch's kernels can be refactored by supplanting their math bits with the corresponding ```attorch.math``` transformations or their derivatives, but doing so would sacrifice the single-file and self-contained design of attorch, so ```attorch.math``` and the rest of attorch will remain separate.

## PyTorch Fallback

To enable easier integration of attorch and PyTorch layers, ```attorch.nn``` is offered, which provides an interface to attorch's modules with PyTorch fallback should a desired layer not be available, as seen below.

```python
from attorch import nn


lin = nn.Linear(10, 20) # Uses attorch's linear layer
gap = nn.AdaptiveAvgPool2d(1) # Uses PyTorch's global pooling since GAP is not available in attorch
```

## Tests

Each module can be tested against its PyTorch counterpart to ensure correctness. These tests are included under ```tests/``` and can be executed using ```pytest```. It should be noted that some might fail owing to numerical precision issues, but in most practical use cases, that should not be a problem.

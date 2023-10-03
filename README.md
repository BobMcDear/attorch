# attorch

attorch is a small subset of PyTorch's neural network modules, written purely in Python using OpenAI's Triton. Its goal is to be easily hackable, self-contained, and readable, whilst maintaining the efficiency and user interface of PyTorch. Currently implemented layers, with full forward and backward pass support, include,

* ```attorch.Conv2d```: 2D-convolves over the input using weights, optionally adding bias.
* ```attorch.GELU```: Applies GELU to the input.
* ```attorch.ReLU```: Applies ReLU to the input.
* ```attorch.SiLU```: Applies SiLU to the input.
* ```attorch.Sigmoid```: Applies sigmoid to the input.
* ```attorch.Tanh```: Applies tanh to the input.
* ```attorch.LogSoftmax```: Normalizes the input using softmax and takes its log.
* ```attorch.Softmax```: Normalizes the input using softmax.
* ```attorch.Softmin```: Normalizes the input using softmin.
* ```attorch.Linear```: Linearly transforms the input using weights, optionally adding bias and fusing an activation function.
* ```attorch.Dropout```: Randomly zeroes elements in the input during training.
* ```attorch.L1Loss```: Measures the mean absolute error between the input and target.
* ```attorch.MSELoss```: Measures the mean squared error between the input and target.

Unless otherwise noted in their docstring, the aforementioned layers behave identically to their PyTorch equivalents.

# Installation

The only dependencies of attorch are ```torch``` and ```triton```. Please install the latest versions of these two libraries and clone this repository to get started.

# Tests

Each module is accompanied by tests that ensure its correctness by testing it against PyTorch layers. They can be conducted by installing ```pytest``` and executing ```pytest```. It should be noted that some tests might fail owing to numerical precision issues, but in most practical use cases, that should not be a problem.

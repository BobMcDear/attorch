"""
Utilities for tests.
"""


from typing import List, Optional, Tuple

import torch
from torch import Tensor

from attorch.types import Device


def default_shapes(min_dim: int = 0, max_dim: int = 4) -> List[Tuple[int, ...]]:
    """
    Returns typical data shapes for testing.

    Args:
        min_dim: Minimum dimensionality of the shapes returned.
        max_dim: Maximum dimensionality of the shapes returned.
    """
    shapes = [(96,),
              (128,),
              (196,),
              (384,),
              (768,),
              (1024,),
              (3200,),
              (4800,),
              (8000,),
              (12288,),
              (1, 8000),
              (4, 2000),
              (8, 1024),
              (32, 1024),
              (128, 1024),
              (2048, 768),
              (6144, 256),
              (8096, 32),
              (12288, 1),
              (1, 1024, 3072),
              (8, 960, 196),
              (64, 768, 128),
              (128, 960, 196),
              (2048, 64, 16),
              (1, 3, 224, 224),
              (8, 3, 224, 224),
              (64, 64, 56, 56),
              (256, 128, 28, 28),
              (256, 2048, 7, 7)]
    return list(filter(lambda shape: min_dim <= len(shape) <= max_dim, shapes))


def create_input(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Device = 'cuda',
    requires_grad: bool = True,
    seed: Optional[int] = 0,
    ) -> Tensor:
    """
    Creates a tensor filled with random numbers.

    Args:
        shape: Shape of tensor.
        dtype: Dtype of tensor.
        device: Device of tensor.
        requires_grad: Flag for recording operations for autodiff.
        seed: Seed for generating random numbers. If None, no seed is set.

    Returns:
        Tensor with random numbers.
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.randn(shape, dtype=dtype, device=device,
                       requires_grad=requires_grad)


def create_input_like(
    input: Tensor,
    requires_grad: bool = False,
    seed: Optional[int] = 0,
    ) -> Tensor:
    """
    Creates a tensor filled with random numbers with the same size, dtype, and
    device as input.

    Args:
        input: Input.
        requires_grad: Flag for recording operations for autodiff.
        seed: Seed for generating random numbers. If None, no seed is set.

    Returns:
        Tensor with random numbers.
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.randn_like(input, requires_grad=requires_grad)


def assert_close(
    *tensor_pairs: Tuple[Tensor, Tensor],
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    ) -> None:
    """
    Asserts that the two tensors in each pair of tensor_pairs are close.
    See also torch.testing.assert_close.

    Args:
        *tensor_pairs: Pairs of tensors that are asserted to be close.
        rtol: Relative tolerance. If specified, atol must also be specified.
            Otherwise, it is selected according to the tensors' dtypes.
        atol: Absolute tolerance. If specified, rtol must also be specified.
            Otherwise, it is selected according to the tensors' dtypes.
    """
    for pair in tensor_pairs:
        torch.testing.assert_close(*pair, rtol=rtol, atol=atol)

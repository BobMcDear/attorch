"""
Utilities for attorch kernels and layers.
"""


from typing import List, Optional

import torch
import triton


def get_output_dtype(
    input_dtype: torch.dtype = torch.float32,
    autocast: Optional[str] = None,
    ) -> torch.dtype:
    """
    Returns the appropriate output dtype for automatic mixed precision
    given the input dtype and the operation's autocast behaviour.

    Args:
        input_dtype: Input dtype.
        autocast: The relevent operation's autocast behaviour.
            None signifies the input dtype should flow through,
            'fp16' signifies autocasting to FP16 when AMP is enabled,
            and 'fp32' signifies autocasting to FP32 when AMP is enabled.
    """
    assert torch.get_autocast_gpu_dtype(), \
        f'Only autocast to float16 is supported, received {torch.get_autocast_gpu_dtype()}'

    if torch.is_autocast_enabled():
        if autocast is None:
            return input_dtype

        elif autocast == 'fp16':
            return torch.float16

        elif autocast == 'fp32':
            return torch.float32

        else:
            raise RuntimeError(f'Autocast type {autocast} is invalid. '
                               'Options are None, fp16, and fp32')

    else:
        return input_dtype


def element_wise_kernel_configs(
    block_name: str = 'BLOCK_SIZE',
    ) -> List[triton.Config]:
    """
    Returns kernel configurations for element-wise operations.

    Args:
        block_name: Name of block argument rows are distributed over.
    """
    return [triton.Config({block_name: 64}, num_warps=2),
            triton.Config({block_name: 128}, num_warps=2),
            triton.Config({block_name: 256}, num_warps=4),
            triton.Config({block_name: 512}, num_warps=4),
            triton.Config({block_name: 1024}, num_warps=4)]


def warps_kernel_configs() -> List[triton.Config]:
    """
    Returns kernel configurations with all possible number of warps.
    """
    return [triton.Config({}, num_warps=2**i) for i in range(6)]

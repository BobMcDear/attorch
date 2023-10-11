"""
Utilities for attorch kernels and layers.
"""


from typing import List

import triton


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

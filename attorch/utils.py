"""
Utilities for attorch kernels and layers.
"""


from typing import List

import triton


def element_wise_kernel_configs() -> List[triton.Config]:
    """
    Returns kernel configurations for element-wise operations.
    """
    return [triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4)]

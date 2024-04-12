"""
Multi-headed attention with PyTorch autodiff support.
"""


from functools import partial
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from triton.ops import attention

from .types import Device


def extract_heads(input: Tensor, num_heads: int) -> Tensor:
    """
    Reshapes the projected input to extract heads for multi-headed attention.

    Args:
        input: Input to reshape and extract heads from.
        num_heads: Number of heads to extract.

    Returns:
        Input reshaped with its heads extracted.
    """
    batch_dim, n_tokens, _ = input.shape
    return input.reshape(batch_dim, n_tokens, num_heads, -1).transpose(1, 2)


class MultiheadAttention(nn.MultiheadAttention):
    """
    Applies multi-headed scaled dot-product attention to the inputs.
    See also base class.

    Args:
        embed_dim: Dimensionality of the query, key, and value inputs.
        num_heads: Number of heads.
        dropout: Dropout probability on the attention scores,
            currently not supported.
        bias: Flag for adding bias to the query-key-value-output projections.
        add_bias_kv: Flag for appending a bias vector to the key and value sequences,
            currently not supported.
        add_zero_attn: Flag for appending a zero vector to the key and value sequences,
            currently not supported.
        kdim: Dimensionality of the key input, which has to be None or equal to embed_dim.
        vdim: Dimensionality of the value input, which has to be None or equal to embed_dim.
        batch_first: Flag to indicate if the batch dimension comes first in the input,
            currently otherwise is not supported.
        device: Device to use.
        dtype: Dtype of layer.

    Raises:
        RuntimeError: 1. Dropout on the attention scores is requested.
                      2. Appending a bias vector to the key and value sequences is requested.
                      3. Appending a zero vector to the key and value sequences is requested.
                      4. The query and key dimensionalities are unequal.
                      5. The query and value dimensionalities are unequal.
                      6. The input is not batch-first.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        if dropout > 0.0:
            raise RuntimeError('Dropout on the attention scores is not supported.')

        if add_bias_kv:
            raise RuntimeError('Appending a bias vector to the key and value '
                               'sequences is not supported.')

        if add_zero_attn:
            raise RuntimeError('Appending a zero vector to the key and value '
                               'sequences is not supported.')

        if kdim is not None and kdim != embed_dim:
            raise RuntimeError(f'The key dimensionality ({kdim}) is not equal to '
                               f'the query dimensionality ({embed_dim}).')

        if vdim is not None and vdim != embed_dim:
            raise RuntimeError(f'The value dimensionality ({vdim}) is not equal to '
                               f'the query dimensionality ({embed_dim}).')

        if not batch_first:
            raise RuntimeError('The input must be batch-first.')

        super().__init__(embed_dim, num_heads, dropout, bias,
                         add_bias_kv, add_zero_attn,
                         kdim, vdim, batch_first,
                         device, dtype)

    def forward(
        self,
        input_q: Tensor,
        input_k: Optional[Tensor] = None,
        input_v: Optional[Tensor] = None,
        causal: bool = False,
        sequence_parallel: bool = False,
        ) -> Tensor:
        assert input_v is None or input_k is not None, \
            f'Key inputs must be provided if value inputs have been passed'

        input_k = input_q if input_k is None else input_k
        input_v = input_k if input_v is None else input_v

        if input_k is input_v:
            if input_q is input_k:
                qkv = F.linear(input_q, self.in_proj_weight, self.in_proj_bias)
                q, k, v = map(partial(extract_heads, num_heads=self.num_heads),
                              qkv.chunk(3, dim=-1))

            else:
                weight_q, weight_kv = torch.split(self.in_proj_weight,
                                                  [self.embed_dim, 2 * self.embed_dim],
                                                  dim=0)

                if self.in_proj_bias is None:
                    bias_q = bias_kv = None

                else:
                    bias_q, bias_kv = torch.split(self.in_proj_bias,
                                                  [self.embed_dim, 2 * self.embed_dim],
                                                  dim=0)

                q = F.linear(input_q, weight_q, bias_q)
                kv = F.linear(input_k, weight_kv, bias_kv)
                q, k, v = map(partial(extract_heads, num_heads=self.num_heads),
                              [q, *kv.chunk(2, dim=-1)])

        else:
            weight_q, weight_k, weight_v = torch.split(self.in_proj_weight,
                                                       3*[self.embed_dim],
                                                       dim=0)

            if self.in_proj_bias is None:
                bias_q = bias_k = bias_v = None

            else:
                bias_q, bias_k, bias_v = torch.split(self.in_proj_bias,
                                                     3*[self.embed_dim],
                                                     dim=0)

            q = F.linear(input_q, weight_q, bias_q)
            k = F.linear(input_k, weight_k, bias_k)
            v = F.linear(input_v, weight_v, bias_v)
            q, k, v = map(partial(extract_heads, num_heads=self.num_heads),
                          [q, k, v])

        output = attention(q, k, v, causal, 0.5, sequence_parallel)
        output = output.transpose(1, 2).reshape(len(input_q), -1, self.embed_dim)
        return self.out_proj(output)

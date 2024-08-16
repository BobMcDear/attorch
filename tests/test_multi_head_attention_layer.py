from typing import Tuple

import pytest
import torch
from torch import autocast, nn
from torch.nn import init

import attorch
from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize('shape', default_shapes(min_dim=3, max_dim=3))
@pytest.mark.parametrize('self_attention', [False, True])
@pytest.mark.parametrize('num_heads', [1, 2, 4, 6, 8])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('amp', [False, True])
def test_multi_head_attention_layer(
    shape: Tuple[int, ...],
    self_attention: bool,
    num_heads: int,
    bias: bool,
    causal: bool,
    input_dtype: bool,
    amp: bool,
    ) -> None:
    if (input_dtype is torch.float16 and not amp) or (shape[-1] % num_heads != 0):
        return

    attorch_input_q = create_input(shape, dtype=input_dtype)
    pytorch_input_q = create_input(shape, dtype=input_dtype)

    if self_attention:
        attorch_input_k = attorch_input_q
        pytorch_input_k = pytorch_input_q

    else:
        attorch_input_k = create_input(shape, dtype=input_dtype, seed=1)
        pytorch_input_k = create_input(shape, dtype=input_dtype, seed=1)

    attorch_input_v = attorch_input_k
    pytorch_input_v = pytorch_input_k

    torch.manual_seed(0)
    attorch_multi_head_attention = attorch.MultiheadAttention(shape[-1], num_heads,
                                                              bias=bias,
                                                              batch_first=True)

    torch.manual_seed(0)
    pytorch_multi_head_attention = nn.MultiheadAttention(shape[-1], num_heads,
                                                         bias=bias,
                                                         batch_first=True,
                                                         device='cuda')

    if bias:
        torch.manual_seed(0)
        init.normal_(attorch_multi_head_attention.in_proj_bias)
        init.normal_(attorch_multi_head_attention.out_proj.bias)

        torch.manual_seed(0)
        init.normal_(pytorch_multi_head_attention.in_proj_bias)
        init.normal_(pytorch_multi_head_attention.out_proj.bias)

    with autocast('cuda', enabled=amp):
        attorch_output = attorch_multi_head_attention(attorch_input_q,
                                                      attorch_input_k,
                                                      attorch_input_v,
                                                      causal=causal)
        pytorch_output = pytorch_multi_head_attention(pytorch_input_q,
                                                      pytorch_input_k,
                                                      pytorch_input_v,
                                                      attn_mask=torch.empty(2*(shape[1],)) if causal else None,
                                                      is_causal=causal,
                                                      need_weights=False)[0]

    assert_close((attorch_output, pytorch_output),
                 rtol=1e-2, atol=1e-2)

    attorch_output.backward(create_input_like(attorch_output))
    pytorch_output.backward(create_input_like(attorch_output))

    bias_grad_pair = ((attorch_multi_head_attention.in_proj_bias,
                       attorch_multi_head_attention.in_proj_bias),
                      (attorch_multi_head_attention.out_proj.bias,
                       attorch_multi_head_attention.out_proj.bias)
                      if bias else ((None, None), (None, None)))
    assert_close((attorch_input_q.grad, pytorch_input_q.grad),
                 (attorch_input_k.grad, pytorch_input_k.grad),
                 (attorch_input_v.grad, pytorch_input_v.grad),
                 (attorch_multi_head_attention.in_proj_weight.grad,
                  pytorch_multi_head_attention.in_proj_weight.grad),
                 (attorch_multi_head_attention.out_proj.weight.grad,
                  pytorch_multi_head_attention.out_proj.weight.grad),
                 *bias_grad_pair,
                 rtol=1e-2, atol=1e-2)

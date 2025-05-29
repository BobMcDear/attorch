"""
GPT2 for language modelling.
"""


from typing import Optional

import torch
from torch import Tensor
from torch import nn

import attorch


class MLP(nn.Module):
    """
    Transforms the input using a multilayer perceptron with one hidden layer
    and the GELU activation function.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        in_dim: Number of input features.
        hidden_dim: Number of hidden features.
        out_dim: Number of output features.
            If None, it is set to the number of input features.
    """
    def __init__(
        self,
        use_attorch: bool,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        ) -> None:
        super().__init__()

        self.fc1 = (attorch.Linear(in_dim, hidden_dim, act_func='gelu')
                    if use_attorch else nn.Linear(in_dim, hidden_dim))
        self.act = nn.Identity() if use_attorch else nn.GELU()
        self.fc2 = (attorch.Linear(hidden_dim, out_dim or in_dim)
                    if use_attorch else nn.Linear(hidden_dim, out_dim or in_dim))

    def forward(self, input: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(input)))


class TransformerBlock(nn.Module):
    """
    Passes the input through a transformer block.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        dim: Embedding dimension.
        num_heads: Number of heads for multi-headed self-attention.
    """
    def __init__(
        self,
        use_attorch: bool,
        dim: int,
        num_heads: int,
        ) -> None:
        super().__init__()
        self.use_attorch = use_attorch
        backend = attorch if use_attorch else nn

        self.ln1 = backend.LayerNorm(dim)
        self.attn = backend.MultiheadAttention(dim, num_heads,
                                               batch_first=True)

        self.ln2 = backend.LayerNorm(dim)
        self.mlp = MLP(use_attorch, dim, 4 * dim)

    def forward(self, input: Tensor) -> Tensor:
        if self.use_attorch:
            output = input + self.attn(self.ln1(input), causal=True)

        else:
            output = self.ln1(input)
            output = input + self.attn(output, output, output,
                                       attn_mask=torch.empty(2*(input.shape[1],)),
                                       is_causal=True,
                                       need_weights=False)[0]

        output = output + self.mlp(self.ln2(output))
        return output


class GPT2(nn.Module):
    """
    Performs language modelling using GPT2,
    optionally computing the loss if return_loss is True.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        vocab_size: Vocabulary size.
        depth: Depth of the transformer.
        dim: Embedding dimension.
        num_heads: Number of heads for multi-headed self-attention.
        max_seq_len: Maximum sequence length of the incoming inputs.
    """
    def __init__(
        self,
        use_attorch: bool,
        vocab_size: int,
        depth: int,
        dim: int,
        num_heads: int,
        max_seq_len: int = 512,
        ) -> None:
        super().__init__()
        backend = attorch if use_attorch else nn

        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.transformer = nn.Sequential(*[TransformerBlock(use_attorch, dim, num_heads)
                                           for _ in range(depth)])
        self.norm = backend.LayerNorm(dim)
        self.fc = backend.Linear(dim, vocab_size)
        self.loss_func = backend.CrossEntropyLoss()

    def forward(
        self,
        input: Tensor,
        return_loss: bool = False,
        ) -> Tensor:
        tok_embed = self.tok_embed(input)
        pos_embed = self.pos_embed(torch.arange(0, input.shape[1],
                                                dtype=torch.long,
                                                device=input.device))

        output = self.transformer(tok_embed + pos_embed)
        output = self.norm(output)
        output = self.fc(output)

        return (self.loss_func(output[:, :-1].contiguous().view(-1, output.shape[-1]),
                               input[:, 1:].contiguous().view(-1))
                if return_loss else output)


def gpt2(
    use_attorch: bool,
    vocab_size: int,
    max_seq_len: 512,
    downsize: int = 1,
    ) -> GPT2:
    """
    Returns a GPT2 model with optional cross entropy loss.

    Args:
        use_attorch: Flag to use attorch in lieu of PyTorch as the backend.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length of the incoming inputs.
        downsize: The depth and width of the model are calculated by dividing
            GPT2's original depth and width by this factor.
    """
    return GPT2(use_attorch, vocab_size=vocab_size,
                depth=12 // downsize, dim=768 // downsize, num_heads=12 // downsize,
                max_seq_len=max_seq_len)

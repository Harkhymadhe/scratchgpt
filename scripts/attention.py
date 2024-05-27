# -*- coding: utf-8 -*-

import torch
from torch import nn

__all__ = [
    "Attention",
    "MultiHeadAttention",
]


class Attention(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.attention_scores = None

    @classmethod
    def get_alignment_vectors(cls, key, query):
        alignment_vectors = torch.matmul(query, key.transpose(-2, -1))
        alignment_vectors /= key.size(-1) ** 0.5

        return alignment_vectors

    def forward(self, key, value, query=None, mask=None):
        if query is None:
            query = key.clone()

        alignment_vectors = self.get_alignment_vectors(key, query)

        if mask:
            alignment_vectors = mask * alignment_vectors

        attention_scores = torch.softmax(alignment_vectors, dim=-1)
        self.attention_scores = attention_scores.detach()

        return torch.matmul(attention_scores, value)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.attention_scores = None

        self.narrow = narrow

        self.query_transform = nn.Linear(
            in_features=self.state_dim, out_features=self.hidden_dim, bias=True
        )

        self.attention_heads = [
            Attention(
            )
            for _ in range(num_heads)
        ]

        self.attention_heads = nn.ModuleList(self.attention_heads)

        if not narrow:
            self.context_transform = nn.Linear(
                in_features=sum([hidden_dim for _ in range(num_heads)]),
                out_features=self.hidden_dim,
                bias=True,
            )

        else:
            # self.query_transform = nn.LazyLinear(out_features=int(hidden_dim * 0.5))

            self.context_transform = nn.Linear(
                in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True
            )

    def get_attention_scores(self):
        self.attention_scores = torch.stack(
            [head.attention_scores for head in self.attention_heads], dim=1
        )
        return

    def get_dims_per_head(self, x):
        N, L, H = x.shape
        x = x.view(N, L, self.num_heads, int(H / self.num_heads))
        return x.transpose(1, 2)

    def forward(self, key, value, query = None, mask=None):
        # query = self.query_transform(query) if self.narrow else query
        # context_vectors = [head(query, mask=mask) for head in self.attention_heads]

        if query is None:
            query = key.clone()

        if self.narrow:  # Implement narrow attention with aggregated context vectors per head
            key = self.get_dims_per_head(key)
            value = self.get_dims_per_head(value)
            query = self.get_dims_per_head(self.query_transform(query))

            batch_size, num_heads, seq_len, new_hidden_dim = query.shape

            # N, n, L, h = query.shape
            # context_vectors = (
            #     self.attention_heads(query, mask=mask).view(N, L, n * h).contiguous()
            # )

            context_vectors = torch.stack(
                [head(key, value, query, mask=mask) for head in self.attention_heads],
                dim = -1
            )

            new_shape = batch_size, seq_len, num_heads * new_hidden_dim
            context_vectors = context_vectors.sum(dim = -1).view(new_shape).contiguous()

        else:  # Implement wide attention with concatenated context vectors per head
            queries = [query] * self.num_heads

            context_vectors = [
                head(key, value, qry, mask=mask) for head, qry in zip(self.attention_heads, queries)
            ]

            context_vectors = torch.concat(context_vectors, dim=-1)

        return self.context_transform(context_vectors)


# if __name__ == "__main__":
#     narrow = False
#     hidden_dim = 128
#     state_dim = 128
#     num_heads = 32
#
#     batch_size = 32
#     seq_len = 10
#
#     attn = MultiHeadAttention(
#         narrow=narrow,
#         hidden_dim=hidden_dim,
#         state_dim=state_dim,
#         num_heads=num_heads,
#     )
#
#     x = torch.randn(batch_size, seq_len, state_dim)
#
#     out = attn(x, x, x, mask=None)
#     print(out.shape)

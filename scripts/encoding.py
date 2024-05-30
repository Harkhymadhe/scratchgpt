# -*- coding: utf-8 -*-

import torch
from torch import nn

__all__ = [
    "TokenEmbedder",
    "EmbeddingSystem",
]


class TokenEmbedder(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding_layer(x)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim=128, max_length=1024):
        super().__init__()

        self.max_length = max_length
        self.hidden_dim = hidden_dim

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-torch.log(torch.tensor([10000.0])) / hidden_dim))

        pos_encoding = torch.zeros(max_length, hidden_dim)

        print(pos_encoding.shape)
        print(torch.sin(position * div_term).shape)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))

        self.dropout = nn.Dropout1d(p = .3)

    def forward(self, x):
        print("POS shape", self.pos_encoding[:, :x.size(1), :].shape)
        return self.dropout(x + self.pos_encoding[:, :x.size(1), :])


class EmbeddingSystem(nn.Module):
    def __init__(
        self,
        num_text_embeddings=1024,
        text_embedding_dim=128,
        num_pos_embeddings=1024,
        pos_embedding_dim=128,
    ):
        super().__init__()
        self.encoder_embedding = TokenEmbedder(num_text_embeddings, text_embedding_dim)
        self.encoder_pos_embedding = PositionalEncoding(
            max_length=num_pos_embeddings,
            hidden_dim=pos_embedding_dim
        )

    def forward(self, x):
        num_pos_embeddings = self.encoder_pos_embedding.max_length

        if x.shape[-1] > num_pos_embeddings:
            x = x[..., :num_pos_embeddings]

        x = self.encoder_embedding(x)
        print(x.shape)
        x = self.encoder_pos_embedding(
            x
        )
        return x

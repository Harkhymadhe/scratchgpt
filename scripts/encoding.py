# -*- coding: utf-8 -*-

import torch
from torch import nn

__all__ = [
    "TokenEncoding",
    "PositionalEncodingv1",
    "PositionalEncodingv2",
    "PositionalEncoding",
    "EmbeddingSystem",
]


class TokenEncoding(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding_layer(x)


class PositionalEncodingv1(nn.Module):
    def __init__(self, hidden_dim=128, max_length=1024, dropout_rate=0.1):
        super().__init__()

        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2)
            * (-torch.log(torch.tensor([10000.0])) / hidden_dim)
        )

        pos_encoding = torch.zeros(max_length, hidden_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

        self.dropout = nn.Dropout1d(p=dropout_rate)

    def forward(self, x):
        return self.dropout(self.pos_encoding[:, : x.size(1), :])


class PositionalEncodingv2(nn.Module):
    def __init__(self, hidden_dim=128, max_length=1024, dropout_rate=0.1):
        super().__init__()

        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.positional_encoding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=hidden_dim
        )

        self.dropout = nn.Dropout1d(p=dropout_rate)

    def forward(self, x):
        x_ = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        return self.dropout(self.positional_encoding(x_))


class PositionalEncoding(nn.Module):
    def __init__(
        self, hidden_dim=128, max_length=1024, dropout_rate=0.1, trainable=True
    ):
        super().__init__()

        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.trainable = trainable

        if trainable:
            self.positional_encoder = PositionalEncodingv2(
                hidden_dim=hidden_dim,
                max_length=max_length,
                dropout_rate=dropout_rate,
            )
        else:
            self.positional_encoder = PositionalEncodingv1(
                hidden_dim=hidden_dim,
                max_length=max_length,
                dropout_rate=dropout_rate,
            )

    def forward(self, x):
        return self.positional_encoder(x)


class EmbeddingSystem(nn.Module):
    def __init__(
        self,
        num_text_embeddings=1024,
        text_embedding_dim=128,
        num_pos_embeddings=1024,
        pos_embedding_dim=128,
        trainable_positional_embeddings=True,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.encoder_embedding = TokenEncoding(num_text_embeddings, text_embedding_dim)
        self.encoder_pos_embedding = PositionalEncoding(
            max_length=num_pos_embeddings,
            hidden_dim=pos_embedding_dim,
            dropout_rate=dropout_rate,
            trainable=trainable_positional_embeddings,
        )

    def forward(self, x):
        num_pos_embeddings = self.encoder_pos_embedding.max_length

        if x.shape[-1] > num_pos_embeddings:
            x = x[..., :num_pos_embeddings]

        x1 = self.encoder_embedding(x)
        x2 = self.encoder_pos_embedding(x)

        return x1 + x2

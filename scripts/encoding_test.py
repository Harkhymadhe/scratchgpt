# -*- coding: utf-8 -*-

import torch
from encoding import PositionalEncoding, TokenEncoding, EmbeddingSystem

if __name__ == "__main__":
    max_length = 1024
    hidden_dim = 128
    num_embeddings = 3000

    x = torch.randint(low=0, high=num_embeddings, size=(8, max_length))
    m1 = PositionalEncoding(
        hidden_dim=hidden_dim, max_length=max_length, trainable=True
    )
    m2 = TokenEncoding(num_embeddings=num_embeddings, embedding_dim=hidden_dim)
    m3 = EmbeddingSystem(
        num_text_embeddings=num_embeddings,
        num_pos_embeddings=max_length,
        pos_embedding_dim=hidden_dim,
        text_embedding_dim=hidden_dim,
        trainable_positional_embeddings=True,
    )

    print(m1(x).shape)
    print(m2(x).shape)

    print((m1(x) + m2(x)).shape)
    print(m3(x).shape)

# -*- coding: utf-8 -*-

import torch
from attention import Attention, MultiHeadAttention


if __name__ == "__main__":
    narrow = True
    hidden_dim = 512
    state_dim = 128
    num_heads = 8

    batch_size = 32
    seq_len = 10

    attn = Attention()

    multi_attn = MultiHeadAttention(
        narrow=narrow,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        num_heads=num_heads,
    )

    x = torch.randn(batch_size, seq_len, state_dim)
    mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)

    out = attn(key=x, value=x, query=x, mask=mask)
    multi_out = multi_attn(key=x, value=x, query=x, mask=mask)

    print(out.shape)
    print(attn.attention_scores.shape)
    print(attn.attention_scores[0].to(torch.float16))

    print("=" * 200 + "\n")
    #
    # print(multi_out.shape)
    #
    # print(multi_attn.attention_heads[-1].attention_scores.shape)
    # print(multi_attn.attention_scores.shape)
    #
    # print("=" * 100 + "\n")
    #
    print(mask)
    # print((mask @ attn.attention_scores).shape)
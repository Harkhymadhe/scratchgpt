# -*- coding: utf-8 -*-

import torch
from torchinfo import summary
from architectures import Transformer

if __name__ == "__main__":

    narrow = True
    hidden_dim = 128
    state_dim = 128
    num_heads = 32
    decoder_only = False
    ablate = False
    pre_ln = True

    batch_size = 32
    seq_len = 10

    attn = Transformer(
        narrow=narrow,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        num_heads=num_heads,
        decoder_only=decoder_only,
        ablate=ablate,
        pre_ln=pre_ln,
    )

    x = torch.randn(batch_size, seq_len, state_dim)

    out = attn(x, x)
    print(out.shape)
    summary(attn, input_data=[x, x])

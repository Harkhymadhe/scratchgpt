# -*- coding: utf-8 -*-

import torch
from llms import PretrainedGPT, GPT, GPTPretrainedHead, GPTHead
from torchinfo import summary

if __name__ == "__main__":

    narrow = True
    hidden_dim = 128
    state_dim = 128
    num_heads = 32
    decoder_only = False
    pre_ln = True

    batch_size = 32
    seq_len = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    attn = PretrainedGPT(
        narrow=narrow,
        num_heads=32,
        num_decoder_blocks=4,
        num_embeddings=3007,
        embedding_dim=hidden_dim,
        max_token=10,
        pre_ln=pre_ln,
        share_weights=True,
    ).to(device)

    # x = torch.randn(batch_size, seq_len, attn.embedding_dim).to(device)
    x = torch.randint(low=0, high=200, size=(batch_size, seq_len)).to(device)

    out = attn(x)
    print(out.shape)

    # head = GPTHead(hidden_dim=seq_len * hidden_dim, vocab=200).to(device)
    # out = head(out)
    # print(out.shape)

    summary(attn, input_data=x)

    print(f"Number of parameters: {sum([p.numel() for p in attn.parameters()])}")

    print(attn.target_mask)
    print("Num decoder blocks:", len(attn.decoder.decoder.decoder))
    print(attn.decoder.decoder.decoder[-1].masked_attention.attention_scores[0, 0])

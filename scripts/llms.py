# -*- coding: utf-8 -*-

import torch
from torch import nn

from architectures import Transformer
from encoding import EmbeddingSystem

__all__ = [
    "GPT",
    "GPTHead",
]


class GPT(nn.Module):
    """Class definition for the Generative Pretrained Model II (GPT-2) backbone."""

    def __init__(
        self,
        num_heads=96,
        num_embeddings=50257,
        max_token=2048,
        embedding_dim=12288,
        num_decoder_blocks=96,
        narrow=True,
        pre_ln=False,
        ablate=True,
    ):
        super(GPT, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_decoder_blocks = num_decoder_blocks
        self.narrow = narrow
        self.num_heads = num_heads

        self.embedding_system = EmbeddingSystem(
            text_embedding_dim=embedding_dim,
            pos_embedding_dim=embedding_dim,
            num_text_embeddings=num_embeddings,
            num_pos_embeddings=max_token,
        )

        self.decoder = Transformer(
            num_decoder_blocks=self.num_decoder_blocks,
            decoder_only=True,
            narrow=self.narrow,
            hidden_dim=self.embedding_dim,
            state_dim=self.embedding_dim,
            num_heads=self.num_heads,
            ablate=ablate,
            pre_ln=pre_ln,
        )

    def forward(self, x, source_mask=None, target_mask=None):
        x_new = self.embedding_system(x)

        x_new = self.decoder(
            x_new, x_new, source_mask=source_mask, target_mask=target_mask
        )

        return x_new


class GPTHead(nn.Module):
    """Class definition for the Generative Pretrained Model II (GPT-2) prediction head"""

    def __init__(self, hidden_dim=128, vocab=50257):
        super(GPTHead, self).__init__()
        self.decoder_head = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=int(hidden_dim * 4), bias=True
            ),
            nn.ReLU(),
            nn.Linear(in_features=int(hidden_dim * 4), out_features=vocab, bias=True),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, -1) if len(x.shape) > 2 else x.view(1, -1)
        x = self.decoder_head(x)

        return x


if __name__ == "__main__":

    from torchinfo import summary

    narrow = True
    hidden_dim = 128
    state_dim = 128
    num_heads = 32
    decoder_only = False
    ablate = False
    pre_ln = True

    batch_size = 32
    seq_len = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn = GPT(
        narrow=narrow,
        num_heads=32,
        num_decoder_blocks=4,
        num_embeddings=3007,
        embedding_dim=hidden_dim,
        ablate=ablate,
        max_token=1024,
        pre_ln=pre_ln,
    ).to(device)

    # x = torch.randn(batch_size, seq_len, attn.embedding_dim).to(device)
    x = torch.randint(low=0, high=200, size=(batch_size, seq_len)).to(device)

    out = attn(x)
    print(out.shape)

    head = GPTHead(hidden_dim=seq_len * hidden_dim, vocab=200).to(device)
    out = head(out)
    print(out.shape)

    summary(attn, input_data=x)

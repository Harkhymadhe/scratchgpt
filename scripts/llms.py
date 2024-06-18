# -*- coding: utf-8 -*-

import torch
from torch import nn

from architectures import Transformer
from encoding import EmbeddingSystem

__all__ = [
    "GPT",
    "GPTHead",
    "GPTPretrainedHead",
    "PretrainedGPT",
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
        dropout_rate=0.1,
        narrow=True,
        pre_ln=False,
        train_pwe=False,
    ):
        super(GPT, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_decoder_blocks = num_decoder_blocks
        self.narrow = narrow
        self.num_heads = num_heads
        self.train_pwe = train_pwe
        self.dropout_rate = dropout_rate

        self.embedding_system = EmbeddingSystem(
            text_embedding_dim=embedding_dim,
            pos_embedding_dim=embedding_dim,
            num_text_embeddings=num_embeddings,
            num_pos_embeddings=max_token,
            trainable_positional_embeddings=train_pwe,
            dropout_rate=dropout_rate,
        )

        self.decoder = Transformer(
            num_decoder_blocks=self.num_decoder_blocks,
            decoder_only=True,
            narrow=self.narrow,
            hidden_dim=self.embedding_dim,
            state_dim=self.embedding_dim,
            num_heads=self.num_heads,
            ablate=True,
            pre_ln=pre_ln,
            dropout_rate=self.dropout_rate,
        )

        attn_shape = (max_token, max_token)
        self.register_buffer(
            "target_mask", torch.tril(torch.ones(attn_shape), diagonal=0)
        )

    def forward(self, x, source_mask=None, target_mask=None):
        x_new = self.embedding_system(x)

        if target_mask is None:
            target_mask = self.target_mask

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


class GPTPretrainedHead(nn.Module):
    """Class definition for the Generative Pretrained Model II (GPT-2) pretraining head"""

    def __init__(self, hidden_dim=128, output_dim=100):
        super(GPTPretrainedHead, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.decoder_head = nn.Linear(
            in_features=hidden_dim, out_features=output_dim, bias=True
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.decoder_head(x)
        return x


class PretrainedGPT(nn.Module):
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
        train_pwe=False,
        share_weights=False,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.gpt_stem = GPT(
            num_heads=num_heads,
            num_embeddings=num_embeddings,
            max_token=max_token,
            embedding_dim=embedding_dim,
            num_decoder_blocks=num_decoder_blocks,
            narrow=narrow,
            pre_ln=pre_ln,
            train_pwe=train_pwe,
            dropout_rate=dropout_rate,
        )

        self.pretrained_head = GPTPretrainedHead(
            hidden_dim=embedding_dim, output_dim=num_embeddings
        )

        # Initialize model parameters
        self._init_params()

        # Tie embedding and classifier weights if required
        if share_weights:
            self.pretrained_head.decoder_head.weight = (
                self.gpt_stem.embedding_system.encoder_embedding.embedding_layer.weight
            )

    def _init_params(self):
        """Initialize model parameters."""
        std = 0.02
        for name, param in self.named_parameters():
            if "decoder" in name:
                std_ = std * (self.gpt_stem.num_decoder_blocks * 2) ** -0.5
                nn.init.normal_(param, mean=0.0, std=std_)
            elif "_encoding" in name:
                nn.init.normal_(param, mean=0.0, std=std)
            elif "feed_forward" in name:
                nn.init.normal_(param, mean=0.0, std=std)
            elif "decoder_head" in name:
                if "bias" in name:
                    nn.init.zeros_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=std)

        return

    def forward(self, x):
        x = self.gpt_stem(x)
        return self.pretrained_head(x)

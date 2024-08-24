# -*- coding: utf-8 -*-

import torch
from torch import nn

from attention import MultiHeadAttention

__all__ = [
    "TransformerEncoderBlock",
    "TransformerDecoderBlock",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
]


class TransformerEncoderBlock(nn.Module):
    """The basic block for the Encoder in the Transformer architecture."""

    def __init__(
        self,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        pre_ln=False,
        dropout_rate=0.1,
        use_bias=False
    ):
        super().__init__()

        self.pre_ln = pre_ln

        self.attention = MultiHeadAttention(
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_heads=num_heads,
            use_bias=use_bias,
            dropout_probability=dropout_rate,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=int(hidden_dim * 4), bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=int(hidden_dim * 4), out_features=hidden_dim, bias=True
            ),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, source_mask=None):
        if self.pre_ln:
            x_ = self.layer_norm1(x)
            x = x + self.dropout1(
                self.attention(key=x_, value=x_, query=x_, mask=source_mask)
            )
            x = x + self.dropout2(self.feed_forward(self.layer_norm2(x)))
        else:
            x = self.layer_norm1(
                x
                + self.dropout1(
                    self.attention(key=x, value=x, query=x, mask=source_mask)
                )
            )
            x = self.layer_norm2(x + self.dropout2(self.feed_forward(x)))

        return x


class TransformerDecoderBlock(nn.Module):
    """The basic block for the Decoder in the Transformer architecture."""

    def __init__(
        self,
        narrow=True,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        pre_ln=True,
        dropout_rate=0.1,
        ablate=False,
        use_bias=False,
    ):
        super().__init__()

        self.pre_ln = pre_ln

        self.ablate = ablate

        if not ablate:
            self.cross_attention = MultiHeadAttention(
                narrow=narrow,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
                use_bias=use_bias,
                dropout_probability=dropout_rate,
            )
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.dropout2 = nn.Dropout(dropout_rate)

        self.masked_attention = MultiHeadAttention(
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_heads=num_heads,
            use_bias=use_bias,
            dropout_probability=dropout_rate,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=int(hidden_dim * 4), bias=True
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(
                in_features=int(hidden_dim * 4), out_features=hidden_dim, bias=True
            ),
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_states, source_mask=None, target_mask=None):

        if self.pre_ln:
            x_ = self.layer_norm1(x)
            x = x + self.dropout1(
                self.masked_attention(key=x_, value=x_, query=x_, mask=target_mask)
            )

            if not self.ablate:
                x = x + self.dropout2(
                    self.cross_attention(
                        key=encoder_states,
                        value=encoder_states,
                        query=self.layer_norm2(x),
                        mask=source_mask,
                    )
                )

            x = x + self.dropout3(self.feed_forward(self.layer_norm3(x)))
        else:
            x = self.layer_norm1(
                x
                + self.dropout1(
                    self.masked_attention(key=x, value=x, query=x, mask=target_mask)
                )
            )

            if not self.ablate:
                x = self.layer_norm2(
                    x
                    + self.dropout2(
                        self.cross_attention(
                            key=encoder_states,
                            value=encoder_states,
                            query=x,
                            mask=source_mask,
                        )
                    )
                )

            x = self.layer_norm3(x + self.dropout3(self.feed_forward(x)))

        return x


class TransformerEncoder(nn.Module):
    """Encoder module for the Transformer architecture."""

    def __init__(
        self,
        num_encoder_blocks=4,
        narrow=True,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        pre_ln=False,
        dropout_rate=0.1,
        use_bias=False,
    ):
        super(TransformerEncoder, self).__init__()

        self.encoder = [
            TransformerEncoderBlock(
                narrow=narrow,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
                pre_ln=pre_ln,
                dropout_rate=dropout_rate,
                use_bias=use_bias,
            )
            for _ in range(num_encoder_blocks)
        ]
        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, x_encoder, source_mask=None):
        for encoder_block in self.encoder:
            x_encoder = encoder_block(x_encoder, source_mask=source_mask)

        return x_encoder


class TransformerDecoder(nn.Module):
    """Decoder module for the Transformer architecture."""

    def __init__(
        self,
        num_decoder_blocks=4,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        ablate=True,
        pre_ln=False,
        dropout_rate=0.1,
        use_bias=False,
    ):
        super(TransformerDecoder, self).__init__()

        self.decoder = [
            TransformerDecoderBlock(
                narrow=narrow,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
                ablate=ablate,
                pre_ln=pre_ln,
                dropout_rate=dropout_rate,
                use_bias=use_bias,
            )
            for _ in range(num_decoder_blocks)
        ]

        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x_decoder, x_encoder, source_mask=None, target_mask=None):
        for decoder_block in self.decoder:
            x_decoder = decoder_block(
                x_decoder, x_encoder, source_mask=source_mask, target_mask=target_mask
            )

        return x_decoder


class Transformer(nn.Module):
    """Class definition for the Transformer architecture."""

    def __init__(
        self,
        num_encoder_blocks=4,
        num_decoder_blocks=4,
        narrow=True,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        decoder_only=False,
        ablate=True,
        pre_ln=False,
        dropout_rate=0.1,
        use_bias=False,
    ):
        super(Transformer, self).__init__()

        self.decoder_only = decoder_only

        if not self.decoder_only:
            self.encoder = TransformerEncoder(
                narrow=narrow,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
                num_encoder_blocks=num_encoder_blocks,
                pre_ln=pre_ln,
                dropout_rate=dropout_rate,
                use_bias=use_bias,
            )

        self.decoder = TransformerDecoder(
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_heads=num_heads,
            num_decoder_blocks=num_decoder_blocks,
            ablate=ablate,
            pre_ln=pre_ln,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
        )

    def forward(self, x_encoder, x_decoder, source_mask=None, target_mask=None):
        if not self.decoder_only:
            x_encoder = self.encoder(x_encoder, source_mask=source_mask)

        x_decoder = self.decoder(
            x_decoder, x_encoder, source_mask=source_mask, target_mask=target_mask
        )

        return x_decoder

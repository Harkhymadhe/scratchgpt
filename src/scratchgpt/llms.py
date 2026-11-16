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
        use_bias=False,
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
        self.use_bias = use_bias

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
            use_bias=use_bias,
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
        use_bias=False,
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
            use_bias=use_bias,
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

    @classmethod
    def _load_weights(cls, model_type="gpt2"):
        model_types = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        assert model_type in model_types, "`model_type` must be one of {}".format(
            ", ".join(model_types[:-1]) + "and " + model_types[-1]
        )
        from transformers import GPT2LMHeadModel

        pretrained_gpt = GPT2LMHeadModel.from_pretrained(model_type)
        return pretrained_gpt

    @classmethod
    def _map_weights_to_layers(
        cls,
        pretrained_model,
        pre_ln=True,
        narrow=True,
        dropout_rate=0.5,
        share_weights=True,
    ):
        print(pretrained_model.config)
        num_heads = pretrained_model.config.n_head
        num_layers = pretrained_model.config.n_layer
        num_embeddings = pretrained_model.config.vocab_size
        max_token = pretrained_model.config.n_ctx
        embedding_dim = pretrained_model.config.n_embd

        model = PretrainedGPT(
            num_heads=num_heads,
            num_embeddings=num_embeddings,
            max_token=max_token,
            embedding_dim=embedding_dim,
            num_decoder_blocks=num_layers,
            narrow=narrow,
            pre_ln=pre_ln,
            train_pwe=True,
            share_weights=share_weights,
            dropout_rate=dropout_rate,
            use_bias=False,
        )

        pretrained_gpt = pretrained_model.state_dict()
        pretrained_gpt = {k: nn.Parameter(v) for k, v in pretrained_gpt.items()}

        ### Load weights for embedding system (token and positional embeddings)

        model.gpt_stem.embedding_system.encoder_embedding.embedding_layer.weight = (
            pretrained_gpt["transformer.wte.weight"]
        )
        model.gpt_stem.embedding_system.encoder_pos_embedding.positional_encoder.positional_encoding.weight = pretrained_gpt[
            "transformer.wpe.weight"
        ]

        ### Load weights for each decoder block
        for layer_index in range(num_layers):
            # Extract decoder block
            model_block = model.gpt_stem.decoder.decoder.decoder[layer_index]

            # Load weights and biases for attention LayerNorm
            model_block.layer_norm1.weight = pretrained_gpt[
                f"transformer.h.{layer_index}.ln_1.weight"
            ]
            model_block.layer_norm1.bias = pretrained_gpt[
                f"transformer.h.{layer_index}.ln_1.bias"
            ]

            # Load weights and biases for FFN LayerNorm
            model_block.layer_norm3.weight = pretrained_gpt[
                f"transformer.h.{layer_index}.ln_2.weight"
            ]
            model_block.layer_norm3.bias = pretrained_gpt[
                f"transformer.h.{layer_index}.ln_2.bias"
            ]

            # Split token transforms into key, query and value weights/biases
            query_bias, key_bias, value_bias = torch.split(
                pretrained_gpt[f"transformer.h.{layer_index}.attn.c_attn.bias"],
                split_size_or_sections=embedding_dim,
                dim=-1,
            )
            query_weight, key_weight, value_weight = torch.split(
                pretrained_gpt[f"transformer.h.{layer_index}.attn.c_attn.weight"],
                split_size_or_sections=embedding_dim,
                dim=-1,
            )

            # Assign weights/biases for key transform to model
            model_block.masked_attention.key_transform.weight = nn.Parameter(
                key_weight.t()
            )
            model_block.masked_attention.key_transform.bias = nn.Parameter(key_bias)

            # Assign weights/biases for query transform to model
            model_block.masked_attention.query_transform.weight = nn.Parameter(
                query_weight.t()
            )
            model_block.masked_attention.query_transform.bias = nn.Parameter(query_bias)

            # Assign weights/biases for value transform to model
            model_block.masked_attention.value_transform.weight = nn.Parameter(
                value_weight.t()
            )
            model_block.masked_attention.value_transform.bias = nn.Parameter(value_bias)

            # Load weights and biases for attended context transform
            model_block.masked_attention.context_transform.weight = nn.Parameter(
                pretrained_gpt[f"transformer.h.{layer_index}.attn.c_proj.weight"].t()
            )
            model_block.masked_attention.context_transform.bias = pretrained_gpt[
                f"transformer.h.{layer_index}.attn.c_proj.bias"
            ]

            # model_block.dropout1 = gpt2_block.attn.attn_dropout
            model_block.feed_forward[0].weight = nn.Parameter(
                pretrained_gpt[f"transformer.h.{layer_index}.mlp.c_fc.weight"].t()
            )
            model_block.feed_forward[0].bias = pretrained_gpt[
                f"transformer.h.{layer_index}.mlp.c_fc.bias"
            ]

            model_block.feed_forward[-1].weight = nn.Parameter(
                pretrained_gpt[f"transformer.h.{layer_index}.mlp.c_proj.weight"].t()
            )
            model_block.feed_forward[-1].bias = pretrained_gpt[
                f"transformer.h.{layer_index}.mlp.c_proj.bias"
            ]

            model.gpt_stem.decoder.decoder.decoder[layer_index] = model_block

        ### Load weights for final layer norm before final output layer
        model.pretrained_head.layer_norm.weight = pretrained_gpt[
            "transformer.ln_f.weight"
        ]
        model.pretrained_head.layer_norm.bias = pretrained_gpt["transformer.ln_f.bias"]

        ### Load weights for final token prediction head
        # TODO: Use this...
        if share_weights:
            model.pretrained_head.decoder_head.weight = model.gpt_stem.embedding_system.encoder_embedding.embedding_layer.weight
        else:
            # TODO: ...or this
            model.pretrained_head.decoder_head.weight = nn.Parameter(
                pretrained_gpt["lm_head.weight"]
            )

            model.pretrained_head.decoder_head.bias = nn.Parameter(
                torch.zeros(pretrained_gpt["lm_head.weight"].shape[0])
            )

        return model

    @classmethod
    def from_weights(
        cls,
        model_type="gpt2",
        pre_ln=True,
        narrow=True,
        dropout_rate=0.5,
        share_weights=False,
    ):
        pretrained_model = cls._load_weights(model_type=model_type)
        return cls._map_weights_to_layers(
            pretrained_model=pretrained_model,
            pre_ln=pre_ln,
            narrow=narrow,
            dropout_rate=dropout_rate,
            share_weights=share_weights,
        )

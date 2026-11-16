# -*- coding: utf-8 -*-

import numpy as np
import torch

from attention import MultiHeadAttention

from architectures import (
    Transformer,
    TransformerDecoder,
    TransformerEncoder,
)
from llms import GPT

from torchinfo import summary

# Runnable tests
tests = [
    "all",
    "attention",
    "transencoder",
    "transdecoder",
    "transformer",
    "gpt1",
    "gpt2",
    "gpt3",
]

test_names = [
    "Attention modules",
    "Transformer-based Encoder architecture",
    "Transformer-based Decoder architecture",
    "Transformer architecture",
    "GPT-1 architecture",
    "GPT-2 architecture",
    "GPT-3 architecture",
]

test_map = {k: v for (k, v) in zip(tests[1:], test_names)}


def run_tests(test_to_run=None, device="cpu"):
    hidden_dim = 160
    num_heads = 5
    narrow = True
    pre_ln = True
    ablate = True

    x = torch.randn(size=(32, 5, hidden_dim)).to(device)
    query = torch.randn(size=(32, 10, hidden_dim)).to(device)

    torch.cuda.empty_cache()

    if test_to_run == "transencoder":
        print("Testing Transformer Encoder\n")

        attn = TransformerEncoder(
            num_heads=num_heads,
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=hidden_dim,
        ).to(device)

        ans = attn(query)

        print(summary(attn, input_data=query, device=device))

        # print("Attention scores shape: ", attn.attention_scores.shape)
        print("Query shape: ", query.shape, end="\n\n")
        print("Transformer Encoder states shape: ", ans.shape, end="\n\n")

    elif test_to_run == "attention":
        print("Testing attention...\n")

        attn = MultiHeadAttention(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            state_dim=hidden_dim,
            narrow=narrow,
        ).to(device)

        ans = attn(query)

        print(summary(attn, input_data=query, device=device))

        print("Attention scores shape: ", attn.attention_scores.shape)
        print("Context vector shape: ", ans.shape, end="\n\n")

    elif test_to_run == "transdecoder":
        print("Testing Transformer Decoder\n")

        attn = TransformerDecoder(
            num_heads=num_heads,
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=hidden_dim,
        ).to(device)

        ans = attn(x_encoder=query, x_decoder=x)

        print(summary(attn, input_data=[query, x], device=device))

        print("Transformer Decoder predictions shape: ", ans.shape, end="\n\n")

    elif test_to_run == "transformer":
        prob = torch.randn(size=(1,))

        if prob >= 0.5:
            print("Using Encoder-Decoder Transformer architecture...\n")
            decoder_only = False
        else:
            print("Using Transformer-Decoder architecture (No Encoder)...\n")
            decoder_only = True

        transformer = Transformer(
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=hidden_dim,
            decoder_only=decoder_only,
            pre_ln=pre_ln,
        ).to(device)

        ans = transformer(x_encoder=x, x_decoder=query)

        print(
            summary(
                transformer,
                input_data=[x, query],
                depth=4,
                batch_dim=None,
                device=device,
            )
        )

        print("=======================================")
        print("States shape: ", x.shape)
        print("Query shape: ", query.shape)
        print("=======================================")
        print("Transformer output shape: ", ans.shape)
        print("`decoder_only` : ", decoder_only)

    elif test_to_run in [
        "gpt1",
        "gpt2",
        "gpt3",
    ]:
        model_constructor = GPT
        if test_to_run == "gpt1":
            num_encoder_embeddings = 50257
            max_token = 1024
            narrow = True
            num_heads = 12
            num_blocks = 12
            embedding_dim = 768
        elif test_to_run == "gpt2":
            num_encoder_embeddings = 50257
            max_token = 2048
            narrow = True
            num_heads = 16
            num_blocks = 48
            embedding_dim = 1600
        elif test_to_run == "gpt3":
            num_encoder_embeddings = int(50257 / 3)
            max_token = 4096
            narrow = True
            num_heads = 12
            num_blocks = int(96 / 3)
            embedding_dim = int(12288 / 3)

        train_pwe = True
        use_bias = True

        # Generate batch of sequences
        num_sequences = 2  # Batch size
        num_tokens = 10  # Number of timesteps per sequence
        x = torch.randint(
            low=0,
            high=num_encoder_embeddings,
            size=(
                num_sequences,
                max_token,
            ),
        ).to(device)
        # my_transformer = Transformer(
        #     narrow=narrow,
        #     hidden_dim=hidden_dim,
        #     state_dim=state_dim,
        #     num_heads=num_heads,
        #     dropout_rate=dropout,
        #     use_bias=use_bias,
        #     ablate=True,
        #     pre_ln=True,
        #     decoder_only=True,
        #     num_decoder_blocks=n_layers
        # )

        model = model_constructor(
            num_heads=num_heads,
            num_decoder_blocks=num_blocks,
            embedding_dim=embedding_dim,
            num_embeddings=num_encoder_embeddings,
            max_token=max_token,
            narrow=narrow,
            pre_ln=pre_ln,
            train_pwe=train_pwe,
            use_bias=use_bias,
        ).to(device)

        ans = torch.sum(model(x), dim=1)

        print(
            summary(
                model,
                input_data=x,
                depth=6,
                batch_dim=None,
                device=device,
            )
        )

        print(f"{test_map[test_to_run].split(' ')[0]} prediction shape: ", ans.shape)

    return


if __name__ == "__main__":
    test = input("Which script to test: ").lower()

    while test not in tests:
        test = input(
            "Option provided must be in [`all`, `attention`, `encoder`, `decoder`, `encdec`, `transformer`, `gpt3`]:  "
        ).lower()

    device = input("\nWhich device to run on: ").lower()
    while device not in ["cuda", "cpu"]:
        device = input("Option provided must be in [`cuda`, `cpu`]:  ").lower()

    print(f"\n{device.upper()} device selected!\n")
    device = torch.device(device)

    if test != "all":
        run_tests(test_to_run=test, device=device)
    else:
        passed_tests = list()
        failed_tests = list()

        for test in tests[1:]:
            if test == "gpt3":
                continue
            print("\n", "=" * 200)
            print(f"\nTesting {test_map[test]}...\n")
            try:
                run_tests(test_to_run=test, device=device)
                passed_tests.append(test)
                torch.cuda.empty_cache()
            except:
                failed_tests.append(test)
                torch.cuda.empty_cache()
                continue

        print("\n", "=" * 200)
        if bool(failed_tests):
            print(
                f"Tests passed: {len(passed_tests)}/{len(passed_tests) + len(failed_tests)}\n"
            )

            print("Passed tests: ")
            for test in passed_tests:
                print(f" > {test}")

            if failed_tests:
                print("\nFailed tests: ")
                for test in failed_tests:
                    print(f" > {test}")
        else:
            print(" " * 90, "All tests run successfully!\n")

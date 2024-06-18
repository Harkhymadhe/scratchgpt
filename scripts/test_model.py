# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import json

from scripts.llms import PretrainedGPT
from scripts.data_ops import DataProcessor

__all__ = [
    "load_state_dicts",
]


def load_state_dicts(timestamp, subfolder=1, tokenizer_path="works_of_fela.json"):
    folder = f"../artefacts/models/{timestamp}/{subfolder}"
    file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    file_paths.sort()

    state_dict_path, _, model_config_path = file_paths
    print(file_paths)

    model_kwargs = dict(json.load(open(model_config_path, "r")))

    tokenizer = DataProcessor(tokenizer_path=tokenizer_path)

    state_dict = torch.load(state_dict_path, map_location="cpu")

    model_state_dict = state_dict["model_state"]
    # print(model_state_dict)

    for k, v in model_kwargs.items():
        print(k, type(v), v)

    model = PretrainedGPT(**model_kwargs)

    model.load_state_dict(state_dict=model_state_dict)

    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()

    return model, tokenizer


def generate_predictions(tokens, model, last_timestep=True):
    model.eval()

    # Obtain model predictions and extract the embeddings at the last timestep
    logits = model(tokens)

    return logits[:, -1:, :] if last_timestep else logits


def temperature_scaling(logits, temperature=1.0):
    """Apply temperature scaling to the model logits."""
    # Scale logits with temperature
    logits /= temperature

    return logits


def multiple_choice_token_selection(
    logits,
    top_k=5,
):
    probabilities = torch.softmax(logits, dim=-1).detach().squeeze()

    top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k)

    logits.squeeze_()
    logits[~top_k_indices] = -torch.inf

    return logits


def probabilistic_sampling(logits):
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next


def generate_text(
    sequence,
    model,
    temperature=1.0,
    top_k=5,
    special_tokens=None,
    length_limit=100,
    post_process=False,
    prob_sampling=True,
    last_timestep=True,
):
    final_pred_tokens = []

    input_tokens = tokenizer.tokenizer.encode(
        sequence=sequence, add_special_tokens=True
    ).ids

    LEN = model.gpt_stem.embedding_system.encoder_pos_embedding.max_length

    if special_tokens is None:
        special_tokens = [
            "<s>",
            "</s>",
            "<pad>",
        ]

    model.eval()

    while len(final_pred_tokens) < length_limit:
        if len(input_tokens) < LEN:
            padded_tokens = [0 for _ in range(LEN - len(input_tokens))] + input_tokens
        else:
            padded_tokens = input_tokens.copy()

        # Convert token ids to tensor
        padded_tokens = torch.tensor(padded_tokens).unsqueeze(0)

        # Generate predicted logits
        logits = generate_predictions(
            tokens=padded_tokens, model=model, last_timestep=last_timestep
        )

        if not last_timestep:
            pred_ = (
                torch.softmax(logits, dim=-1)
                .max(dim=-1)
                .indices.squeeze()
                .numpy()
                .tolist()
            )
            print(tokenizer.tokenizer.decode(ids=pred_))
            logits = logits[:, -1:, :]

        # Apply top-k sampling
        selected_logits = multiple_choice_token_selection(logits, top_k)

        # Apply temperature scaling
        scaled_logits = temperature_scaling(
            logits=selected_logits, temperature=temperature
        )

        # Apply probabilistic sampling
        if prob_sampling:
            pred_tokens = probabilistic_sampling(scaled_logits)
        else:
            pred_tokens = torch.softmax(scaled_logits, dim=-1).argmax(dim=-1).item()

        pred_tokens = pred_tokens if isinstance(pred_tokens, list) else [pred_tokens]

        # print(pred_tokens)
        # print(tokenizer.tokenizer.decode(ids=pred_tokens))

        final_pred_tokens += pred_tokens
        pred_token = tokenizer.tokenizer.decode(ids=pred_tokens[-1:])

        input_tokens += pred_tokens
        input_tokens = input_tokens[1:]

        if pred_token in special_tokens:
            if pred_token == "</s>":
                print("End of generation!")
                break
            continue

    pred_text = tokenizer.tokenizer.decode(ids=final_pred_tokens)

    if post_process:
        pred_text = pred_text.replace("\n ", "\n")

    return {
        "original_text": sequence,
        "predicted_text": pred_text,
        "combined_text": sequence + pred_text,
    }


if __name__ == "__main__":
    timestamp = "20240614"
    subfolder = 2

    model, tokenizer = load_state_dicts(timestamp=timestamp, subfolder=subfolder)

    test_sequence_a = "Something must kill man o\nSomething must kill man o\nWhether na water, we no know o"
    test_sequence_b = "Something must kill man\nSomething must kill man o"
    test_sequence_c = (
        "Trouble dey sleep, yanga go wake am\nTrouble dey sleep, yanga go wake am\n"
        "Who send you, I neva know o"
    )
    test_sequence_d = "Please generate some songs by Fela:"
    test_sequence_e = """Them go dabaru everything
Them go turn green into white
Them go turn red into blue
Water dey go, water dey come
Water dey go, water dey come
Them go turn electric to candle
Them go turn electric to candle
Government magic
Government magic"""

    temperature = 1.0
    top_k = 50
    length_limit = 100
    post_process = False
    prob_sampling = True
    last_timestep = True

    pred_text = generate_text(
        sequence=test_sequence_c.lower(),
        model=model,
        temperature=temperature,
        top_k=top_k,
        length_limit=length_limit,
        post_process=post_process,
        prob_sampling=prob_sampling,
        last_timestep=last_timestep,
    )
    print(pred_text)

    print("=" * 100)
    print(pred_text["combined_text"])
    print("=" * 100)

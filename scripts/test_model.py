# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import json

from collections import OrderedDict

from scripts.llms import PretrainedGPT
from scripts.data_ops_v1 import DataProcessor

from peft import LoraConfig, get_peft_model

__all__ = [
    "load_state_dicts",
]


def load_state_dicts(timestamp, subfolder=1, tokenizer_path="works_of_fela.json", lora=False):
    """Load the trained model for use."""

    if lora:
        tokenizer_path = "gpt2"

        model = PretrainedGPT._load_weights(tokenizer_path)
        for param in model.parameters():
            param.requires_grad = False

        # LoRa
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, config)
    else:

        folder = f"../artefacts/models/{timestamp}/{subfolder}"
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        file_paths.sort()

        state_dict_path, _, model_config_path = file_paths
        print(file_paths)

        model_kwargs = dict(json.load(open(model_config_path, "r")))

        state_dict = torch.load(state_dict_path, map_location="cpu")

        model_state_dict = state_dict["model_state"]

        for n in model_state_dict:
            print(n)

        # print(model_state_dict)

        for k, v in model_kwargs.items():
            print(k, type(v), v)


        model = PretrainedGPT(**model_kwargs)

        try:
            model.load_state_dict(state_dict=model_state_dict)
        except:
            model_state_dict = refactor_state_dict(model_state_dict)
            model.load_state_dict(state_dict=model_state_dict)

    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    tokenizer = DataProcessor(tokenizer_path)

    return model, tokenizer


def refactor_state_dict(model_state_dict):
    old_key = "gpt_stem.embedding_system.encoder_pos_embedding.positional_encoder.pos_encoding"
    new_key = "gpt_stem.embedding_system.encoder_pos_embedding.positional_encoder.positional_encoding.weight"

    model_state_dict = OrderedDict(
        ((k, v) if old_key != k else (new_key, v.squeeze()))
        for k, v in model_state_dict.items()
    )
    return model_state_dict


def generate_predictions(tokens, model, last_timestep=True):
    """Generate predictions given the input tokens and the model."""

    model.eval()

    # Obtain model predictions and extract the embeddings at the last timestep
    logits = model(tokens)

    try:
        logits = logits["logits"]
    except:
        pass

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
    probs = probs[probs >= 0]
    print("Probs:", probs)
    idx_next = torch.multinomial(probs, num_samples=1)
    print("idx_next:", idx_next)
    return idx_next


def generate_text_v1(
    sequence,
    model,
    tokenizer,
    temperature=1.0,
    top_k=5,
    special_tokens=None,
    length_limit=100,
    post_process=False,
    prob_sampling=True,
    last_timestep=True,
    device=None,
):
    if device is None:
        device = torch.device("cpu")

    final_pred_tokens = []

    try:
        input_tokens = tokenizer.tokenizer.encode(
            sequence=sequence, add_special_tokens=True
        ).ids
    except:
        input_tokens = tokenizer.tokenizer.encode(sequence)

    try:
        LEN = model.gpt_stem.embedding_system.encoder_pos_embedding.max_length
    except:
        LEN = 1024

    if special_tokens is None:
        special_tokens = [
            "<s>",
            "</s>",
            "<pad>",
        ]

    model.eval()
    model.to(device)

    try:
        pad_token = tokenizer.tokenizer.eos_token
    except:
        pad_token = 0

    while len(final_pred_tokens) < length_limit:
        padded_tokens = input_tokens + final_pred_tokens

        if len(padded_tokens) < LEN:
            padded_tokens = [
                pad_token for _ in range(LEN - len(padded_tokens))
            ] + padded_tokens

            print("Number of input tokens now:", len(padded_tokens))
        elif len(padded_tokens) > LEN:
            padded_tokens = padded_tokens[-LEN:]
        else:
            pass

        # Convert token ids to tensor
        padded_tokens = torch.tensor(padded_tokens).unsqueeze(0).to(device)

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
        try:
            pred_token = tokenizer.tokenizer.decode(ids=pred_tokens[-1:])
        except:
            pred_token = tokenizer.tokenizer.decode(tokens=pred_tokens[-1:])

        if pred_token in special_tokens:
            if pred_token == "</s>":
                print("End of generation!")
                break
            continue

    try:
        pred_text = tokenizer.tokenizer.decode(ids=final_pred_tokens)
    except:
        pred_text = tokenizer.tokenizer.decode(tokens=final_pred_tokens)

    if post_process:
        pred_text = pred_text.replace("\n ", "\n")

    return {
        "original_text": sequence,
        "predicted_text": pred_text,
        "combined_text": sequence + pred_text,
    }


def generate_text(
    sequence,
    model,
    tokenizer,
    temperature=1.0,
    top_k=5,
    special_tokens=None,
    length_limit=100,
    post_process=False,
    prob_sampling=True,
    last_timestep=True,
    device=None,
):
    if device is None:
        device = torch.device("cpu")

    final_pred_tokens = []

    try:
        input_tokens = tokenizer.tokenizer.encode(
            sequence=sequence, add_special_tokens=True
        ).ids
    except:
        input_tokens = tokenizer.tokenizer.encode(sequence)

    try:
        LEN = model.gpt_stem.embedding_system.encoder_pos_embedding.max_length
    except:
        LEN = 1024

    if special_tokens is None:
        special_tokens = [
            "<s>",
            "</s>",
            "<pad>",
        ]

    model.eval()
    model.to(device)

    try:
        pad_token = tokenizer.tokenizer.eos_token
    except:
        pad_token = 0

    while len(final_pred_tokens) < length_limit:
        padded_tokens = input_tokens + final_pred_tokens

        if len(padded_tokens) < LEN:
            padded_tokens = [
                pad_token for _ in range(LEN - len(padded_tokens))
            ] + padded_tokens

            print("Number of input tokens now:", len(padded_tokens))
        elif len(padded_tokens) > LEN:
            padded_tokens = padded_tokens[-LEN:]
        else:
            pass

        # Convert token ids to tensor
        padded_tokens = torch.tensor(padded_tokens).unsqueeze(0).to(device)

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

        # Apply softmax
        if temperature > 0:
            logits /= temperature

        logits = torch.softmax(logits, dim=-1)

        # Apply top-k sampling
        top_k_probs, top_k_indices = torch.topk(logits, top_k)

        print("top_k_probs:", top_k_probs.shape)

        idx = torch.multinomial(top_k_probs.squeeze(), 1)

        pred_tokens = torch.gather(top_k_indices.squeeze(), -1, idx).item()

        pred_tokens = pred_tokens if isinstance(pred_tokens, list) else [pred_tokens]

        # print(pred_tokens)
        # print(tokenizer.tokenizer.decode(ids=pred_tokens))

        final_pred_tokens += pred_tokens
        try:
            pred_token = tokenizer.tokenizer.decode(ids=pred_tokens[-1:])
        except:
            pred_token = tokenizer.tokenizer.decode(tokens=pred_tokens[-1:])

        if pred_token in special_tokens:
            if pred_token == "</s>":
                print("End of generation!")
                break
            continue

    try:
        pred_text = tokenizer.tokenizer.decode(ids=final_pred_tokens)
    except:
        pred_text = tokenizer.tokenizer.decode(tokens=final_pred_tokens)

    if post_process:
        pred_text = pred_text.replace("\n ", "\n")

    return {
        "original_text": sequence,
        "predicted_text": pred_text,
        "combined_text": sequence + pred_text,
    }


if __name__ == "__main__":
    SEED = 42

    INFO = {
        "orwell": {
            "tokenizer_path": "george_orwell.json",
            "timestamp": "20240905",
            "subfolder": 1
        },
        "poe": {
            "tokenizer_path": "edgar-alan_poe.json",
            "timestamp": "20240905",
            "subfolder": 2
        }
    }

    key = input("Input one of `orwell` or `poe`: ").lower().strip()
    KEY_DICT = INFO[key]

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.cuda.manual_seed(seed=SEED)
    else:
        DEVICE = torch.device("cpu")
        SEED = torch.manual_seed(SEED)

    lora = False

    if lora:
        timestamp = KEY_DICT['timestamp']
    else:
        timestamp = "20240905" if "poe" in key else "20240827"

    subfolder = KEY_DICT['subfolder']
    tokenizer_path = KEY_DICT['tokenizer_path']

    model, tokenizer = load_state_dicts(
        timestamp=timestamp, subfolder=subfolder, tokenizer_path=tokenizer_path, lora=lora
    )


    alan_poe_context_1 = """How truthful an air of lamentations hangs here upon every syllable! It pervades all."""

    alan_poe_context_2 = "The initial consideration was that of extent. If any literary work is too long to be read at one sitting,"

    alan_poe_context_3 = "It appears evident, then, that there is a distinct limit,"

    alan_poe_context_4 = "By the murmur of a spring,\nOr the least boughs rustleling,"


    orwell_context_1 = """St Athelstan's Church stood at the highest point of Knype Hill, and if you chose to climb the tower you could see ten miles or so across the surrounding country.  Not that there was anything worth looking at--only the low, barely undulating East Anglian landscape, intolerably dull in summer, but redeemed in winter by the recurring patterns of the elms, naked and fanshaped against leaden skies."""

    orwell_context_2 = """Most people who bother with the matter at all would admit that the English language is in a bad way, but it is generally assumed that we cannot by conscious action do anything about it. Our civilization is decadent, and our language--so the argument runs--must inevitably share in the general collapse. It follows that any struggle against the abuse of language is a sentimental archaism, like preferring candles to electric light or hansom cabs to aeroplanes. Underneath this lies the half-conscious belief that language is a natural growth and not an instrument which we shape for our own purposes."""

    orwell_context_3 = """It was part of the economy drive in preparation for Hate Week. The flat was seven flights up, and Winston, who was thirty-nine and had a varicose ulcer above his right ankle, went slowly, resting several times on the way. On each landing, opposite the lift-shaft, the poster with the enormous face gazed from the wall. It was one of those pictures which are so contrived that the eyes follow you about when you move. BIG BROTHER IS WATCHING YOU, the caption beneath it ran."""

    orwell_context_4 = """The pool was swarming with bream, small ones, about four to six inches long. """

    POE_CONTEXTS = [
        alan_poe_context_1,
        alan_poe_context_2,
        alan_poe_context_3,
        alan_poe_context_4,
    ]

    ORWELL_CONTEXTS = [
        orwell_context_1,
        orwell_context_2,
        orwell_context_3,
        orwell_context_4,
    ]

    CONTEXTS = POE_CONTEXTS if "poe" in key else ORWELL_CONTEXTS

    index = 0


    # Hyperparameters for generation
    temperature = 0.5
    top_k = 50
    length_limit = 50
    post_process = False
    prob_sampling = True
    last_timestep = True

    pred_text = generate_text(
        sequence=CONTEXTS[index].lower() if not lora else CONTEXTS[index],
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        top_k=top_k,
        length_limit=length_limit,
        post_process=post_process,
        prob_sampling=prob_sampling,
        last_timestep=last_timestep,
        device=DEVICE,
    )
    print(pred_text)

    print("=" * 100)
    print(pred_text["combined_text"])
    print("=" * 100)

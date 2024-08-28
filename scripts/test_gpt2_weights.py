# -*- coding: utf-8 -*-

import torch
import tiktoken

from scripts.llms import PretrainedGPT
from scripts.data_ops import DataProcessor
from scripts.test_model import generate_text



if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "gpt2"
    model = PretrainedGPT.from_weights(model_type=model_type)
    tokenizer = DataProcessor(tokenizer_path="edgar_alan_poe.json")

    print("""================================================================""")

    print(model)

    gpt2_tokenizer = tiktoken.get_encoding("gpt2")

    print(gpt2_tokenizer.decode([gpt2_tokenizer.eot_token]))
    text = "My name is Oladele, from Nigeria"
    tokens = gpt2_tokenizer.encode(text)

    tokenizer.tokenizer = gpt2_tokenizer

    if len(tokens) < 1024:
        tokens += [gpt2_tokenizer.eot_token for _ in range(1024 - len(tokens))]

    token_tensor = torch.tensor(tokens).unsqueeze(0)

    generated_tokens = model(token_tensor)

    print(generated_tokens)

    test_sequence_a = "Something must kill man o\nSomething must kill man o\nWhether na water, we no know o"
    test_sequence_b = "Something must kill man\nSomething must kill man o"
    test_sequence_c = (
        "Trouble dey sleep, yanga go wake am\nTrouble dey sleep, yanga go wake am\n"
        "Who send you, I neva know o"
    )
    test_sequence_d = "Please generate some songs by Fela:"
    test_sequence_e = """Them go dabaru everything\nThem go turn green into white\nThem go turn red into blue
       "\nWater dey go, water dey come"
       "Water dey go, water dey come\nThem go turn electric to candle\n"
       "Them go turn electric to candle\nGovernment magic\nGovernment magic"""
    test_sequence_f = """And I said to her, "Never let them weigh you down"."""

    temperature = .1
    top_k = 50
    length_limit = 100
    post_process = False
    prob_sampling = True
    last_timestep = True

    pred_text = generate_text(
        sequence="Hello, I'm a language model,",
        model=model,
        tokenizer=tokenizer,
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

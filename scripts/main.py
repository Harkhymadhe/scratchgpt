# -*- coding: utf-8 -*-
import os.path

import torch
from torch.utils.data import DataLoader

from scripts.llms import PretrainedGPT
from scripts.data_ops_v1 import TextDataset
from scripts.train_ops import training_loop_with_batch

from torch import optim

from torchinfo import summary
import json


if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()

    args.add_argument("--files", type=str, default=None)

    args.add_argument("--vocab_size", type=int, default=50000)

    args = args.parse_args()

    if args.files is None:
        file_names = "george_orwell.txt"
    else:
        file_names = args.files

    tokenizer_name = os.path.splitext(file_names)[0] + ".json"

    # Set hyperparameters
    model_kwargs = dict(json.load(open("../config/model_config.json")))
    hyperparameter_kwargs = dict(
        json.load(open("../config/hyperparameters_config.json"))
    )

    narrow = model_kwargs["narrow"]
    num_heads = model_kwargs["num_heads"]
    num_decoder_blocks = model_kwargs["num_decoder_blocks"]
    num_embeddings = model_kwargs["num_embeddings"]
    embedding_dim = model_kwargs["embedding_dim"]
    max_token = model_kwargs["max_token"]
    pre_ln = model_kwargs["pre_ln"]
    share_weights = model_kwargs["share_weights"]
    dropout_rate = model_kwargs["dropout_rate"]
    train_pwe = model_kwargs["train_pwe"]

    seq_len = hyperparameter_kwargs["seq_len"]
    epochs = hyperparameter_kwargs["epochs"]
    lr = hyperparameter_kwargs["lr"]
    batch_size = hyperparameter_kwargs["batch_size"]
    weight_decay = hyperparameter_kwargs["weight_decay"]
    use_last = hyperparameter_kwargs["use_last"]

    batch_size_in_tokens = 2**19
    batch_size = 32
    gradient_accumulation_steps = 10
    # gradient_accumulation_steps = batch_size_in_tokens / (batch_size * seq_len)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    train_frac = 0.85

    assert train_frac <= 1.0, "Ensure the train fraction is between 0 and 1."
    test_frac = 1.0 - train_frac
    augment_data = True

    train_dataset = TextDataset(
        file_name=file_names,
        tokenizer_path=tokenizer_name,
        time_steps=seq_len,
        min_length=seq_len,
        train=True,
        frac=train_frac,
        augment_data=augment_data,
    )

    test_dataset = TextDataset(
        file_name=file_names,
        tokenizer_path=tokenizer_name,
        time_steps=seq_len,
        min_length=seq_len,
        train=False,
        frac=test_frac,
        augment_data=augment_data,
    )

    # Generate dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # Sample data loader
    x_sample, y_sample = next(iter(train_dataloader))
    x_sample, y_sample = x_sample.to(device), y_sample.to(device)

    # Set TF32 precision
    torch.set_float32_matmul_precision("high")

    # Instantiate model
    model = PretrainedGPT(
        narrow=narrow,
        num_heads=num_heads,
        num_decoder_blocks=num_decoder_blocks,
        num_embeddings=train_dataset.data_processor.get_vocab_size(),
        embedding_dim=embedding_dim,
        max_token=max_token,
        pre_ln=pre_ln,
        train_pwe=train_pwe,
        share_weights=share_weights,
        dropout_rate=dropout_rate,
    )
    model.to(device)

    # # Compile model
    # if hasattr(torch, "compile"):
    #     print("Compiling model...")
    #     model = torch.compile(model, backend="inductor")

    # Instantiate optimizer

    decay_params = list(filter(lambda x: x.dim() >= 2, model.parameters()))
    non_decay_params = list(filter(lambda x: x.dim() < 2, model.parameters()))

    num_decay_params = sum(p.numel() for p in decay_params)
    num_non_decay_params = sum(p.numel() for p in non_decay_params)

    print(
        "Number of decaying params: {}\nNumber of non-decaying params: {}".format(
            num_decay_params, num_non_decay_params
        )
    )

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": non_decay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(
        params=param_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),  # beta values for GPT-3
        fused=True,
    )

    # Train model and return results
    # train_artefacts = training_loop_with_batch(
    #     model=model,
    #     train_dataloader=train_dataloader,
    #     test_dataloader=test_dataloader,
    #     optimizer=optimizer,
    #     scheduler=None,
    #     epochs=epochs,
    #     device=device,
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     chained_scheduler=True,
    #     use_last=use_last,  # DO NOT CHANGE!
    # )

    train_artefacts = training_loop_with_batch(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=None,
        epochs=epochs,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        chained_scheduler=False,
        use_scheduler=True,
        resume_training=False,
        use_last=use_last,  # DO NOT CHANGE!
    )

    model_kwargs["num_embeddings"] = train_dataset.data_processor.get_vocab_size()

    # Save config files for future reference
    json.dump(model_kwargs, open(train_artefacts["model_config_path"], "w"))
    json.dump(
        hyperparameter_kwargs, open(train_artefacts["hyperparameter_config_path"], "w")
    )

    train_artefacts["model"] = train_artefacts["model"].eval()

    for param in train_artefacts["model"].parameters():
        param.requires_grad = False

    out = train_artefacts["model"](x_sample)

    summary(train_artefacts["model"], input_data=x_sample)

    print("Number of embeddings:", model_kwargs["num_embeddings"])
    print("Final output shape:", out.shape)

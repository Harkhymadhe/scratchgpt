# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

from scripts.llms import PretrainedGPT
from scripts.data_ops import FelaDataset
from scripts.train_ops import training_loop_with_batch

from torch import optim

from torchinfo import summary


if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()

    args.add_argument("--files", type=str, default=None)

    args.add_argument("--vocab_size", type=int, default=50000)

    args = args.parse_args()

    if args.files is None:
        file_paths = "../datasets/works_of_fela.txt"
    else:
        file_paths = args.files

    epochs = 20
    lr = 1e-3

    narrow = True

    hidden_dim = 64
    state_dim = 64
    max_token = 256
    seq_len = 100

    dataset = FelaDataset(
        data_path=file_paths,
        tokenizer_path="works_of_fela.json",
        time_steps=seq_len,
    )

    num_embeddings = dataset.data_processor.get_vocab_size()
    num_heads = 8
    num_decoder_blocks = 8
    decoder_only = False

    ablate = False
    pre_ln = True

    batch_size = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    x_sample, y_sample = next(iter(dataloader))
    x_sample, y_sample = x_sample.to(device), y_sample.to(device)

    attn = PretrainedGPT(
        narrow=narrow,
        num_heads=num_heads,
        num_decoder_blocks=num_decoder_blocks,
        num_embeddings=num_embeddings,
        embedding_dim=hidden_dim,
        ablate=ablate,
        max_token=max_token,
        pre_ln=pre_ln,
    ).to(device)

    optimizer = optim.AdamW(attn.parameters(), lr=lr, foreach=True)

    train_artefacts = training_loop_with_batch(
        model=attn,
        dataset=dataloader,
        optimizer=optimizer,
        scheduler=None,
        epochs=epochs,
        device=device,
        stack_metrics=False,  # DO NOT CHANGE!
    )

    train_artefacts["model"] = train_artefacts["model"].eval()

    for param in train_artefacts["model"].parameters():
        param.requires_grad = False

    out = train_artefacts["model"](x_sample)

    summary(train_artefacts["model"], input_data=x_sample)

    print("Number of embeddings:", num_embeddings)
    print("Final output shape:", out.shape)

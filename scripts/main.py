# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

from scripts.data_ops import DataTokenizer
from scripts.llms import GPT, GPTHead
from scripts.data_ops import FelaDataset

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
    seq_len = 101

    dataset = FelaDataset(
        data_path=file_paths,
        tokenizer_path="../artefacts/tokenizers/works_of_fela.json",
        time_steps=seq_len
    )

    num_embeddings = dataset.data_processor.get_vocab_size()
    num_heads = 8
    num_decoder_blocks = 8
    decoder_only = False

    ablate = False
    pre_ln = True

    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    x_sample, y_sample = next(iter(dataloader))
    x_sample, y_sample = x_sample.to(device), y_sample.to(device)

    attn = GPT(
        narrow=narrow,
        num_heads=num_heads,
        num_decoder_blocks=num_decoder_blocks,
        num_embeddings=num_embeddings,
        embedding_dim=hidden_dim,
        ablate=ablate,
        max_token=max_token,
        pre_ln=pre_ln,
    ).to(device)

    # x = torch.randn(batch_size, seq_len, attn.embedding_dim).to(device)
    x = torch.randint(low=0, high=200, size=(batch_size, seq_len)).to(device)

    out = attn(x_sample)
    print(out.shape)

    head = GPTHead(hidden_dim=(seq_len-1) * hidden_dim, vocab=200).to(device)
    out = head(out)
    print(out.shape)

    summary(attn, input_data=x)

# -*- coding: utf-8 -*-

import os

import torch
from torch.utils.data import Dataset
from torch import nn

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE

from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from transformers import GPT2TokenizerFast, GPT2Tokenizer

__all__ = ["DataTokenizer", "DataProcessor", "DataProcessorv2", "FelaDataset"]


class DataTokenizer:
    def __init__(
        self,
        file_paths,
        tokenizer=None,
        vocab_size=50000,
        max_length=1024,
        special_tokens=None,
    ):

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            tokenizer = Tokenizer(BPE())

            tokenizer.normalizer = Sequence([Lowercase()])

            tokenizer.pre_tokenizer = ByteLevel()
            tokenizer.decoder = ByteLevelDecoder()

            self.tokenizer = tokenizer

        self.vocab_size = vocab_size

        if isinstance(file_paths, (list, tuple, set)) or file_paths is None:
            self.file_paths = file_paths
        else:
            self.file_paths = [file_paths]

        if special_tokens is None:
            special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

        self.special_tokens = special_tokens
        self.max_length = max_length

        self.tokenizer.enable_padding(
            direction="right",
            pad_token="<pad>",
            pad_id=self.special_tokens.index("<pad>"),
            length=self.max_length,
        )

        # self.tokenizer.enable_truncation(
        #     direction="right",
        #     max_length=self.max_length,
        # )

        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            inital_alphabet=ByteLevel.alphabet(),
            special_tokens=special_tokens,
        )

    def train_tokenizer(self, file_path):
        self.tokenizer.train([file_path], self.trainer)
        self.tokenizer.save(
            f"../artefacts/tokenizers/{os.path.splitext(file_path)[0].split('/')[-1]}.json"
        )
        return

    def train_tokenizers(self, file_paths=None):
        if file_paths is None:
            file_paths = self.file_paths

        for file_path in file_paths:
            self.train_tokenizer(file_path)

        return

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids=ids, **kwargs)

    def encode(self, sequence, **kwargs):
        return self.tokenizer.encode(sequence=sequence, **kwargs)

    def decode_pipeline(self, x, **kwargs):
        encodings = self.encode(x, **kwargs)
        return self.decode(encodings.ids, **kwargs)

    @classmethod
    def from_pretrained(cls, path):
        tokenizer = cls(file_paths=None, tokenizer=Tokenizer.from_file(path))
        return tokenizer


class DataProcessor:
    def __init__(
        self,
        tokenizer_path,
        vocab_size=50000,
    ):
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        self.tokenizer = DataTokenizer.from_pretrained(tokenizer_path)

    def process_file(self, file_path, **kwargs):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.readlines()

        # content = [c.strip() for c in content]
        merged_content = " ".join(content)
        tokenized_content = self.tokenizer.encode(merged_content, **kwargs)

        # sample_len = 100
        # examples = []
        # for i in range(0, len(tokenized_content)):
        #     examples.append(tokenized_content[i:i + sample_len])

        return [e.ids for e in self.tokenizer.tokenizer.encode_batch(content)]


class DataProcessorv2:
    def __init__(
        self,
        tokenizer_path,
        vocab_size=50000,
    ):
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        self.tokenizer = DataTokenizer.from_pretrained(tokenizer_path)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def process_file(self, file_path, time_steps=100, **kwargs):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.readlines()

        # content = [c.strip() for c in content]
        content = " ".join(content)
        tokenized_content = self.tokenizer.encode(content, **kwargs).ids

        self.tokenized_content = tokenized_content

        num_tokens = len(tokenized_content)
        self.num_tokens = num_tokens

        data_samples = [
            tokenized_content[i : i + time_steps]
            for i in range(0, num_tokens)
        ]
        pad_id = self.tokenizer.special_tokens.index("<pad>")
        data_samples[-1] = data_samples[-1] + [pad_id for _ in range(time_steps - len(data_samples[-1] ))]

        data_samples = self.offset_augmentation(data_samples, time_steps = time_steps, offset_ratios = None)
        data_samples = self.overlap_augmentation(data_samples, time_steps=time_steps, overlap_ratios=None)

        return data_samples

    def offset_augmentation(self, data_samples, time_steps = 100, offset_ratios = None):
        if offset_ratios is None:
            offset_ratios = [.2, .3, .4, .5, .6, .7, .8]

        for offset_ratio in offset_ratios:
            start_index = int(self.num_tokens * offset_ratio)

            offset_data_samples = [
                self.tokenized_content[i: i + time_steps]
                for i in range(start_index, self.num_tokens)
            ]

            data_samples.extend(offset_data_samples)

        return data_samples

    def overlap_augmentation(self, data_samples, time_steps = 100, overlap_ratios = None):
        if overlap_ratios is None:
            overlap_ratios = [.2, .3, .4, .5, .6, .7, .8]

        for overlap_ratio in overlap_ratios:
            start_index = int(time_steps * overlap_ratio)

            still_overlapping = True
            i = 0
            overlap_data_samples = []

            while still_overlapping:
                overlap_index = torch.randint(low=0, high=start_index, size = (1,)).item()

                start_overlap_index = i if i == 0 else i - overlap_index
                end_overlap_index = start_overlap_index + time_steps

                print(f"Starting at: {start_overlap_index}")

                if end_overlap_index >= self.num_tokens:
                    print(f"End index: {end_overlap_index} greater than token number: {self.num_tokens}")
                    still_overlapping = False
                    end_overlap_index = self.num_tokens - 1

                sample = self.tokenized_content[start_overlap_index: end_overlap_index]

                if len(sample) < time_steps:
                    pad_id = self.tokenizer.special_tokens.index("<pad>")
                    sample = sample + [pad_id for _ in range(time_steps - len(sample))]

                overlap_data_samples.append(sample)
                i = end_overlap_index

            data_samples.extend(overlap_data_samples)

        return data_samples


class FelaDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer_path,
        time_steps=100,
        vocab_size=50000,
    ):
        super().__init__()
        self.path = data_path
        self.data_processor = DataProcessorv2(
            tokenizer_path,
            vocab_size=vocab_size,
        )
        self.data = self.data_processor.process_file(
            file_path=data_path, time_steps=time_steps
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data[:-1], data[1:]


if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()

    args.add_argument("--files", type=str, default=None)

    args.add_argument("--vocab_size", type=int, default=50000)

    args = args.parse_args()

    if args.files is None:
        file_paths = ["../datasets/works_of_fela.txt"]
    else:
        file_paths = args.files

    tokenizer = DataTokenizer(file_paths=file_paths, vocab_size=args.vocab_size)
    tokenizer.train_tokenizers()

    print(tokenizer.tokenizer.get_vocab_size())
    encoding = tokenizer.tokenizer.encode("No wahala, senior man.")
    decoding = tokenizer.tokenizer.decode(encoding.ids)
    #
    # print(encoding.tokens)
    # print(decoding)

    dataset = FelaDataset(
        data_path=file_paths[0],
        tokenizer_path="../artefacts/tokenizers/works_of_fela.json"
    )

    data, label = dataset[len(dataset)-1]

    print("=" * 50, "Data")

    print(dataset.data_processor.tokenizer.decode(data))

    print("="*50, "Label")
    print(dataset.data_processor.tokenizer.decode(label))
    print("=" * 50)

    token_population = len(dataset) * 100

    factor = 3

    print("Total number of tokens for training:", token_population)
    print("Total number of matching model parameters:", int(token_population/factor))

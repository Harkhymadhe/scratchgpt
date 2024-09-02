# -*- coding: utf-8 -*-

import os
import re
import string

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE

from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

__all__ = ["DataTokenizer", "DataProcessor", "DataProcessorv2", "TextDataset"]


class DataTokenizer:
    def __init__(
        self,
        file_names,
        tokenizer=None,
        vocab_size=50000,
        max_length=1024,
        special_tokens=None,
        enable_padding=False,
        enable_truncation=False,
        use_gpt2_tokenizer=False,
    ) -> None:
        """Tokenization object for tokenizing text corpus.

        Args:
            file_names (list): List of file names.
            tokenizer (Tokenizer): Tokenizer to be used.
            vocab_size (int): Vocab size.
            max_length (int): Max length of tokenized text.
            special_tokens (list): List of special tokens.
            enable_padding (bool): If True, add padding token to tokenized text.
            enable_truncation (bool): If True, truncate tokenized text.

        Return:
            None.
        """

        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.get_vocab_size()
        else:
            tokenizer = Tokenizer(BPE())

            tokenizer.normalizer = Sequence([Lowercase()])

            tokenizer.pre_tokenizer = ByteLevel()
            tokenizer.decoder = ByteLevelDecoder()

            self.tokenizer = tokenizer

            self.vocab_size = vocab_size

        if isinstance(file_names, (list, tuple, set)) or file_names is None:
            self.file_names = file_names
        else:
            self.file_names = [file_names]

        if special_tokens is None:
            special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]

        self.special_tokens = special_tokens
        self.max_length = max_length
        self.enable_padding = enable_padding
        self.enable_truncation = enable_truncation

        self.use_gpt2_tokenizer = use_gpt2_tokenizer

        if enable_padding:
            self.tokenizer.enable_padding(
                direction="right",
                pad_token="<pad>",
                pad_id=self.special_tokens.index("<pad>"),
                length=self.max_length,
            )

        if enable_truncation:
            self.tokenizer.enable_truncation(
                direction="right",
                max_length=self.max_length,
            )

        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            inital_alphabet=ByteLevel.alphabet(),
            special_tokens=special_tokens,
        )

    def train_tokenizer(self, file_name):
        raw_file_path = os.path.join("../datasets/raw", file_name)
        preprocessed_file_path = os.path.join("../datasets/preprocessed", file_name)

        # Preprocess text file
        self.preprocess_file(
            raw_file_path=raw_file_path, preprocessed_file_path=preprocessed_file_path
        )

        self.tokenizer.train([preprocessed_file_path], self.trainer)
        self.tokenizer.save(
            f"../artefacts/tokenizers/{os.path.splitext(preprocessed_file_path)[0].split('/')[-1]}.json"
        )
        return

    def train_tokenizers(self, file_names=None):
        if file_names is None:
            file_names = self.file_names

        for file_name in file_names:
            self.train_tokenizer(file_name)

        return

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(
            ids=ids if isinstance(ids, list) else ids.numpy().tolist(), **kwargs
        )

    def encode(self, sequence, **kwargs):
        return self.tokenizer.encode(sequence=sequence, **kwargs)

    def decode_pipeline(self, x, **kwargs):
        encodings = self.encode(x, **kwargs)
        return self.decode(encodings.ids, **kwargs)

    @classmethod
    def from_pretrained(cls, tokenizer_name):
        if tokenizer_name.replace(".json", "").lower() == "gpt2":
            use_gpt2_tokenizer = True
        else:
            use_gpt2_tokenizer = False

        tokenizer_name = (
            tokenizer_name + ".json"
            if not tokenizer_name.endswith(".json")
            else tokenizer_name
        )
        path = f"../artefacts/tokenizers/{tokenizer_name}"
        tokenizer = cls(
            file_names=os.path.splitext(tokenizer_name)[0],
            tokenizer=Tokenizer.from_file(path),
            use_gpt2_tokenizer=use_gpt2_tokenizer,
        )
        return tokenizer

    @classmethod
    def preprocess_file(cls, raw_file_path, preprocessed_file_path, eos_token="</s>"):
        with open(raw_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        lines = list(filter(lambda x: "※" not in x, lines))

        # re.findall(r"\s*[\*\s]*", line)
        lines = [
            (
                eos_token
                if len(list(set(re.sub("\s*", "", line)))) > 0
                and list(set(re.sub("\s*", "", line)))[0] in ["_", "*"]
                else line
            )
            for line in lines
        ]

        with open(preprocessed_file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return


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
        merged_content = "".join(content)
        tokenized_content = self.tokenizer.encode(merged_content, **kwargs)

        # sample_len = 100
        # examples = []
        # for i in range(0, len(tokenized_content)):
        #     examples.append(tokenized_content[i:i + sample_len])

        return [e.ids for e in self.tokenizer.tokenizer.encode_batch(content)]


class DataProcessorv2:
    def __init__(
        self, tokenizer_name, vocab_size=50000, min_length=20, augment_data=False
    ):
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_name

        self.min_length = min_length

        self.tokenizer = DataTokenizer.from_pretrained(tokenizer_name)
        self.use_gpt2_tokenizer = self.tokenizer.use_gpt2_tokenizer

        if self.use_gpt2_tokenizer:
            self.eos_token = "<|endoftext|>"
            self.bos_token = "<|endoftext|>"
        else:
            self.eos_token = "</s>"
            self.bos_token = "<s>"

        self.tokenized_content = None
        self.num_tokens = None

        self.enable_padding = self.tokenizer.enable_padding
        self.enable_truncation = self.tokenizer.enable_truncation
        self.augment_data = augment_data

    def get_vocab_size(self):
        return self.tokenizer.tokenizer.get_vocab_size()

    @staticmethod
    def load_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.readlines()

        return content

    def preprocess_fela_file(self, file_contents):
        # Filter out lines linking to the website the Fela data was obtained from
        file_contents = list(filter(lambda x: "※" not in x, file_contents))

        song_boundaries = [
            i for i, line in enumerate(file_contents) if list(set(line[:-1])) == ["_"]
        ]

        return file_contents, song_boundaries

    def preprocess_edgar_file(self, file_contents):
        # Filter out lines filled with whitespaces
        file_contents = list(filter(lambda x: len(x.lstrip()) != 0, file_contents))

        song_boundaries = [
            i
            for i, line in enumerate(file_contents)
            if (line.lstrip()[0] == "*" and line.lstrip()[-2] == "*")
        ]

        return file_contents, song_boundaries

    def preprocess_orwell_file(self, file_contents):
        # Filter out lines filled with whitespaces
        file_contents = list(filter(lambda x: len(x.lstrip()) != 0, file_contents))

        song_boundaries = [
            i for i, line in enumerate(file_contents) if line.startswith("THE END")
        ]

        return file_contents, song_boundaries

    def preprocess_line(self, line):
        punctuation = string.punctuation.replace(" ", "")
        punctuation = punctuation.replace(",", "")
        punctuation = punctuation.replace(";", "")

        return (
            line
            if len(line) == 1
            else (line if line[-2] in punctuation else line.strip() + " ")
        )

    def preprocess_file(self, file_contents, file="fela"):
        assert file in [
            "fela",
            "orwell",
            "poe",
        ], "File must be one of `{}`, `{}`, or `{}`".format("fela", "orwell", "poe")
        final_contents = []

        if file == "fela":
            file_contents, song_boundaries = self.preprocess_fela_file(file_contents)
        elif file == "orwell":
            file_contents, song_boundaries = self.preprocess_orwell_file(file_contents)
        else:
            file_contents, song_boundaries = self.preprocess_edgar_file(file_contents)

        start_idx = 0
        for i in song_boundaries:
            common_lines = file_contents[start_idx:i]
            common_lines = list(map(self.preprocess_line, common_lines))
            book = "".join(common_lines)
            final_contents.append(f"{self.bos_token}\n" + book + f"\n{self.eos_token}")
            start_idx = i + 1

        if start_idx < len(file_contents) - 1:
            common_lines = file_contents[start_idx:]
            common_lines = list(map(self.preprocess_line, common_lines))
            book = "".join(common_lines)
            final_contents.append(f"{self.bos_token}\n" + book + f"\n{self.eos_token}")

        return final_contents

    def process_file(self, file_path, time_steps=100, **kwargs):
        file_contents = self.load_file(file_path=file_path)
        file = os.path.splitext(file_path.split(os.sep)[-1])[0].split("_")[-1]

        file_contents = self.preprocess_file(file_contents=file_contents, file=file)

        processed_chunks = []

        for chunk in file_contents:
            processed_chunks += self.process_file_chunk(
                file_chunk=chunk, time_steps=time_steps, **kwargs
            )

        return processed_chunks

    def process_file_chunk(self, file_chunk, time_steps=100, **kwargs):
        tokenized_content = self.tokenizer.encode(file_chunk, **kwargs).ids

        self.tokenized_content = tokenized_content

        num_tokens = len(tokenized_content)
        self.num_tokens = num_tokens

        pad_id = self.tokenizer.special_tokens.index("<pad>")

        data_samples = [
            (
                tokenized_content[i : i + time_steps + 1]
                if len(tokenized_content[i : i + time_steps + 1]) == time_steps + 1
                else tokenized_content[i : i + time_steps + 1]
                + [
                    pad_id
                    for _ in range(
                        time_steps + 1 - len(tokenized_content[i : i + time_steps + 1])
                    )
                ]
            )
            for i in range(0, num_tokens)
        ]

        print(f"Total number of raw chunks: {len(data_samples)}")
        print(f"Size of each raw chunk: {len(data_samples[0])}")

        if self.augment_data:
            data_samples = self.offset_augmentation(
                data_samples, time_steps=time_steps, offset_ratios=None
            )
            data_samples = self.overlap_augmentation(
                data_samples, time_steps=time_steps, overlap_ratios=None
            )

        data_samples = self.filter_data(data_samples, min_length=self.min_length)
        print(f"Total number of tokens: {len(data_samples)}")
        # data_samples = [
        #     (
        #         sample
        #         if len(sample) == time_steps
        #         else sample + [pad_id for _ in range(time_steps - len(sample))]
        #     )
        #     for sample in data_samples
        # ]
        return data_samples

    def offset_augmentation(self, data_samples, time_steps=100, offset_ratios=None):
        if offset_ratios is None:
            offset_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        for offset_ratio in offset_ratios:
            start_index = int(self.num_tokens * offset_ratio)

            pad_id = self.tokenizer.special_tokens.index("<pad>")

            offset_data_samples = [
                (
                    self.tokenized_content[i : i + time_steps + 1]
                    if len(self.tokenized_content[i : i + time_steps + 1])
                    == time_steps + 1
                    else self.tokenized_content[i : i + time_steps + 1]
                    + [
                        pad_id
                        for _ in range(
                            time_steps
                            + 1
                            - len(self.tokenized_content[i : i + time_steps + 1])
                        )
                    ]
                )
                for i in range(start_index, self.num_tokens)
            ]

            data_samples.extend(offset_data_samples)

        print("Data samples generated!")

        return data_samples

    def overlap_augmentation(self, data_samples, time_steps=100, overlap_ratios=None):
        if overlap_ratios is None:
            overlap_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        for overlap_ratio in overlap_ratios:
            start_index = int(time_steps * overlap_ratio)

            still_overlapping = True
            i = 0
            overlap_data_samples = []

            while still_overlapping:
                overlap_index = torch.randint(low=0, high=start_index, size=(1,)).item()

                start_overlap_index = i if i == 0 else i - overlap_index
                end_overlap_index = start_overlap_index + time_steps + 1

                if end_overlap_index >= self.num_tokens:
                    # print(
                    #     f"End index: {end_overlap_index} greater than token number: {self.num_tokens}"
                    # )
                    still_overlapping = False
                    end_overlap_index = self.num_tokens

                sample = self.tokenized_content[start_overlap_index:end_overlap_index]

                if len(sample) < time_steps + 1:
                    pad_id = self.tokenizer.special_tokens.index("<pad>")
                    sample = sample + [
                        pad_id for _ in range(time_steps + 1 - len(sample))
                    ]

                overlap_data_samples.append(sample)
                i = end_overlap_index

            data_samples.extend(overlap_data_samples)

        return data_samples

    def filter_data(self, data_samples, max_pad=0.0, min_length=20):
        pad_id = self.tokenizer.special_tokens.index("<pad>")

        data_samples = list(filter(lambda x: len(x) >= min_length, data_samples))

        # if self.enable_padding:
        data_samples = list(
            filter(lambda x: x.count(pad_id) / len(x) <= max_pad, data_samples)
        )

        return data_samples


class TextDataset(Dataset):
    def __init__(
        self,
        file_name,
        tokenizer_path,
        time_steps=100,
        vocab_size=50000,
        min_length=100,
        frac=0.85,
        train=None,
        augment_data=False,
    ):
        super().__init__()
        self.file_name = file_name
        self.path = os.path.join("../datasets/preprocessed", file_name)
        self.frac = frac
        self.train = train
        self.time_steps = time_steps
        self.augment_data = augment_data

        self.data_processor = DataProcessorv2(
            tokenizer_path,
            vocab_size=vocab_size,
            min_length=min_length,
            augment_data=augment_data,
        )
        self.data = self.data_processor.process_file(
            file_path=self.path, time_steps=time_steps
        )

        if isinstance(self.train, bool):
            end_index = int(self.frac * len(self.data))
            self.data = self.data[:end_index] if self.train else self.data[end_index:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data[:-1], data[1:]


if __name__ == "__main__":
    file_name = "george_orwell"
    file = file_name.split("_")[-1]

    file_path = "../datasets/raw/" + file_name

    processor = DataProcessorv2(tokenizer_name=f"{file_name}.json")
    file_contents = processor.process_file(file_path=f"{file_path}.txt")

    # file_contents_ = processor.preprocess_edgar_file(file_contents=file_contents)
    # file_contents_ = processor.process_file(file_contents=file_contents, file=file)

    print(file_contents)
    print(f"Number of stories by George Orwell: {len(file_contents)}")
    print(file_contents[-1])

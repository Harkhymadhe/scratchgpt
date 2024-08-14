# -*- coding: utf-8 -*-

import os
from data_ops import TextDataset, DataTokenizer, DataProcessorv2

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Obtain hyperparameter values

    args = ArgumentParser()

    args.add_argument("--files", type=str, default=None)

    args.add_argument("--vocab_size", type=int, default=50000)

    args = args.parse_args()

    if args.files is None:
        file_names = "george_orwell.txt"
    else:
        file_names = args.files

    augment_data = False
    train = None

    # Train tokenizer
    tokenizer = DataTokenizer(file_names=file_names, vocab_size=args.vocab_size)
    # tokenizer.train_tokenizers()

    tokenizer = DataTokenizer.from_pretrained(os.path.splitext(file_names)[0])

    print(f"Vocabulary size for dataset: {tokenizer.tokenizer.get_vocab_size()}")
    encoding = tokenizer.tokenizer.encode("No wahala, senior man.")
    decoding = tokenizer.tokenizer.decode(encoding.ids)
    #
    # print(encoding.tokens)
    # print(decoding)

    # Create dataset instance
    dataset = TextDataset(
        file_name=file_names,
        tokenizer_path=os.path.splitext(file_names)[0] + ".json",
        time_steps=256,
        augment_data=augment_data,
        train=train,
    )

    # Sample from the dataset
    data, label = dataset[len(dataset) - 1]

    print("=" * 50, "Data")

    print(data)
    print(dataset.data_processor.tokenizer.decode(data))

    print("=" * 50, "Label")
    print(label)
    print(dataset.data_processor.tokenizer.decode(label))
    print("=" * 50)

    token_population = len(dataset) * dataset.time_steps

    factor = 3
    batch_size = 256

    print(
        "Total number of tokens for training (dataset length by timestep size):",
        token_population,
    )
    print(
        "Total number of tokens for training (number of characters):",
        dataset.data_processor.num_tokens,
    )
    print("Total number of matching model parameters:", int(token_population / factor))
    print(
        "Number of batches with batch size of",
        batch_size,
        ":",
        len(dataset) // batch_size,
    )

    index = 3000
    d1, d2 = dataset[index]

    print(dataset.data_processor.tokenizer.decode(d1).replace("\n", "\\n"))
    print(dataset.data_processor.tokenizer.decode(d2).replace("\n", "\\n"))
    print(d1.shape)

    n = 0

    for i, (d1, d2) in enumerate(dataset):
        if d1.shape != d2.shape:
            print(f"X and y shape mismatch for batch {i}!")
            n += 1
        else:
            # print(f"Time steps should be {dataset.time_steps}")
            if d1.shape[-1] != dataset.time_steps:
                print(
                    f"Wrong time step (should be {dataset.time_steps}) for batch {i}: X = {d1.shape} | y = {d2.shape}"
                )
                n += 1

    print(f"Faulty batches: {n}/{len(dataset)}")
    print(f"Number of batches: {len(dataset)/batch_size}")
    print(f"Data Augmentation: {augment_data}")

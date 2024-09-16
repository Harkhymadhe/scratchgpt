## LLM

---

### Overview

---
This project is an attempt to code up the GPT model family from scratch. This involved coding p the below:

1. Attention mechanism
    
    a. Self- and Cross-attention (&#9989;)

    b. Wide and Narrow attention (&#9989;)
    
    c. Positional encoding (&#9083;, &#8987;)


2. Transformer architecture

    a. Transformer v1 i.e., Encoder-Decoder (&#9989;)

    b. Transformer v2 i.e., Decoder-only (&#9989;)


3. GPT family

   a. GPT-1 (full version &#9989;)
   
   b. GPT-2 (full version &#9989;)
   
   c. GPT-3 (reduced version &#9989;)


The default hyperparameters for the GPT family are as tabulated below:

| Property                 | GPT-1       | GPT-2       | GPT-3       |
|--------------------------|-------------|-------------|-------------|
| Number of parameters     | 117 Million | 1.5 Billion | 175 Billion |
| Number of decoder blocks | 12          | 48          | 96          |
| Embedding dimension      | 768         | 1600        | 12288       |
| Maximum token size       | 1024        | 2048        | 4096        |

### Repository Structure

---

The structure of the repo is as such:

```text
.
├── README.MD
├── main.py (⚠, ⌛)
├── assets
│   └── gpt-arch.png
├── data
│   └── io.py (⚠, ⌛)
├── main.py
└── scripts
    ├── architectures.py
    ├── attention.py
    ├── llms.py
    └── tests.py
```

### Architectural structures

---

***GPT-1***
&#9989;

---
***GPT-2***
&#9989;

---
***GPT-3***

Reduced version &#9989;

<a href="https://dugas.ch/artificial_curiosity/img/GPT_architecture/fullarch.png">![Full size](assets/gpt-arch.png)</a>
<p align="center">Image Credit: <a href="https://dugas.ch/artificial_curiosity/GPT_architecture.html">Dugas' blog</a></p>

---
### Running tests

---

At present, tests can be run for the:
1. Attention (and MultiHeadAttention) implementation (`attention`)
2. Transformer (`transformer`)
3. GPT-1 (`gpt1`)
4. GPT-2 (`gpt2`)
5. GPT-3 (reduced version. See [Note](#note)) (`gpt3`)

The tests can be run like so:

```bash
$ cd scripts
$ python tests.py
```

Next, you provide the name of the test to run. For instance, to run all tests, provide `all`. Select device to run test on as eitehr `cpu` or `cuda`. If the test runs successfully, you should get something like:

```text
Test `test name` succeeded!
```


### Legend

---

| Symbol   | Meaning                                |
|----------|----------------------------------------|
| &#9989;  | Implemented successfully               |
| &#10060; | Implemented; incomplete or failing     |
| &#9083;  | Placeholder in place; work in progress |
| &#9888;  | Excluded file; present in `.gitignore` |
| &#8987;  | Work in progress                       |

### Model Inference

To run inference on the trained models, the following steps are to be followed:

```bash
$ cd scripts
$ python test_model.py
```
This should prompt a selection process for the dataset in question (`edgar` or `poe`), whether to use the LoRA model, 
and what prompt to test on. Finally, select one of four prompts to generate text  with. 
The selected model will then be automatically used to generate completions, which will be printed on screen.

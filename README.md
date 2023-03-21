# Datasets and Code for "TransAM : Transformer Appending Matcher for Few-Shot Knowledge Graph Completion"

## Datasets

- NELL-One (datasets/nell)
- Wiki-One (datasets/wiki)

## To prepare data

Embeddings are from [GMatching](https://github.com/xwhan/One-shot-Relational-Learning) repo. Please place embeddings in `datasets/<data>/embed` dir.

We also provide an option to help obtain pretrain embeddings from sketch with [OpenKE](https://github.com/thunlp/OpenKE) : 

1. Run `python process.py <wiki / nell>` to generate data for OpenKE pretraining. (files in `<dataset>/openke`)
2. Copy files and run [OpenKE](https://github.com/thunlp/OpenKE) traning.
3. Copy pre-trained files back to datasets/<dataset> directory.
4. Specific the filename in `toml` type configuration file (`embed_fn=<openke output>`) 

## Requirements

```requirements.txt
python>=3.10
torch>=1.11
rich
typer
```

## Run

`python trainer.py fit <config>.toml`

Configure file will generate in `toml/` dir.

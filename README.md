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

## Dataset Instructions
Each dataset has following files:

- `ent2ids` json type, key is entity symbol and value is the entity id. Used for load embeddings
- `relation2ids` json type, key is relation symbol and value is the entity id. Used for load embeddings
- `train_tasks.json` json type, key is few-shot relation, value is list of triples, i.e. `[(h, r, t)]`. Used for training.
- `valid_tasks.json` and `dev_tasks.json` json type, key is few-shot relation, value is list of triples (**symbol, not ids**), i.e. `[(h, r, t)]`. Used for validation.
- `test_tasks.json` json type, key is few-shot relation, value is list of triples (**symbol, not ids**), i.e. `[(h, r, t)]`. Used for testing.
- `path_graph` the background graph. Text type. Each line is a triple (**symbol, not ids**), separating head, relation and tail with `\t` symbol.
- `e1rel_e2.json` for negative sampling. Key is `headrel` (symbol head and relation, directly concatenation) , value is list of all positive tails.
- `embed` for storaging embeddings
- `openke` for storaging `OpenKE` type files for obtaining embeddings. (Used in `process.py`)

## Requirements

```requirements.txt
python==3.10
torch>=1.11
rich
typer
```

## Run

`python trainer.py fit <config>.toml`

Configure file will generate in `toml/` dir.

## Argument instructions

```toml
candidate_size = -1  # configure candidate set size. To deal with FSRL issue.
checkpoint = "path/to/checkpoint"  # checkpoint path
dataset = "nell" # dataset, wiki or nell
debug = false  # debug mode
embed_fn = "complEx_nell.ckpt"  # openKE type embedding file
eval_every = 10000  # epoch interval
eval_size = 2000  # batch eval to solve memory overflow issue for some GPUs.
gpu = -1  # set GPU, -1 for using CPU
grad_clip = 5.0  # enable gradient clipping
log_every = 50  # log interval
lr = 5e-5  # learning rate
margin = 2.0  # deprecated field, for margin loss
max_epochs = 500000  # max training epoch
mode = "append"
name = "TransAM"
neg_rate = 1  # negative sampling rate
parent = ""
pretrain = "ComplEx"  # pre-training model
query_size = 100  # query set size
seed = 1234
test = false
test_in_fit = true
warmup_epochs = 10000
weight_decay = 0.0

[model]
activation_fn = "gelu"  # activation function in transformer
aggr_heads = "keep" 
dropout_embed = 0.3  # dropout
dropout_tr = 0.3  # dropout
embed_dim = 100
encoder = "attn"
matcher = "single"
max_neighbors = 50  # max neighbor size
num_heads = 4  # number of transformer heads
num_layers = 3  # number of transformer layers
shot = 5
```

# Fine-Tuning Embedding Model Sample

This repository contains the code for fine-tuning an embedding model.
It is slightly updated/organized from Phil Schmid's [Fine-tune Embedding models for Retrieval Augmented Generation](https://www.philschmid.de/fine-tune-embedding-model-for-rag) blog post.

## Requirements

* Python 3.9 or better
* 3GB VRAM/RAM available (see `BATCH_SIZE` configuration in the *Modifying the Settings* section below)

## Datasets

The test and training datasets are in the `datasets` directory.
You can also find the `create-dataset.py` to create the datasets from scratch.

## Modifying the Settings

There are a few knobs you can turn in `const.py`:
* `EMBEDDING_MODEL_ID`: The Hugginface account/repositor ID of the base embedding model
* `MATRYOSHKA_DIMENSIONS`: The dimensions to use for Matryoshka embeddings, ordered in decreasing order.
* `DEVICE`: The device to use during training and evaluation.
   By default it's `auto`, which uses CUDA or Metal (Apple Silicon) if available, otherwise CPU is used.
* `BATCH_SIZE`: The batch size to use during training.
  You should set this depending on how much VRAM/RAM (if on CPU) can be allocated during training.
  Smaller values will use less VRAM/RAM, but training time will be longer.
  Setting it to `1` will enable running the fine tuning process on 3GB RAM/VRAM with the defaults.
* `TRAIN_EPOCHS`: Number of training epochs.
  
## Evaluating the Baseline

Run `python3 evaluate-model.py` to evaluate the base model (`EMBEDDING_MODEL_ID` in `const.py`).

## Fine-Tuning the Base Model

Run `python3 finetune-model.py` to fine-tune the base model and save it to `output/fine-tuned-embedding-model`.

## Evaluate the Fine-Tuned Model

Run `python3 evaluate-model.py output/fine-tuned-embedding-model` to evaluate the fine-tuned model.

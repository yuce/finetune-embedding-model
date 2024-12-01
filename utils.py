from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator

from datasets import Dataset, load_dataset, concatenate_datasets

from const import DEVICE, MATRYOSHKA_DIMENSIONS

@dataclass
class Datasets:
    test: Dataset
    train: Dataset
    corpus: Dataset


def get_device() -> str:
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.mps.is_available():
            return "mps"
        return "cpu"
    return DEVICE


def load_datasets() -> Datasets:
    current_dir = Path(__file__).parent
    dataset_dir = current_dir / "datasets"
    test_dataset = load_dataset("json", data_files=str(dataset_dir / "test_dataset.json"), split="train")
    train_dataset = load_dataset("json", data_files=str(dataset_dir / "train_dataset.json"), split="train")
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])
    return Datasets(test=test_dataset, train=train_dataset, corpus=corpus_dataset)


def create_matryoshka_evaluator(test_dataset: Dataset, corpus_dataset: Dataset) -> SequentialEvaluator:
    corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
    queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))
    # create a mapping of relevant docs for each query
    relevant_docs = {q_id: {q_id} for q_id in queries}
    # create the evaluator
    matryoshka_evaluators = []
    for dim in MATRYOSHKA_DIMENSIONS:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to the given dimension
        )
        matryoshka_evaluators.append(ir_evaluator)
    return SequentialEvaluator(matryoshka_evaluators)


def output_path(path: str) -> str:
    current_dir = Path(__file__).parent
    return str(current_dir / "output" / path)

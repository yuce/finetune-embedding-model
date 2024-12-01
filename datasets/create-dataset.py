# see: https://www.philschmid.de/fine-tune-embedding-model-for-rag

from pathlib import Path

from datasets import load_dataset

# load the datasets
dataset = load_dataset("philschmid/finanical-rag-embedding-dataset", split="train")

# rename the cols
dataset = dataset.rename_column("question", "anchor")
dataset = dataset.rename_column("context", "positive")

# add an id columnn
dataset = dataset.add_column("id", range(len(dataset)))

# split datasets into a 10% test set
dataset = dataset.train_test_split(test_size=0.1)

current_dir = Path(__file__).parent

# save datasets to disk
dataset["train"].to_json(current_dir / "train_dataset.json", orient="records")
dataset["test"].to_json(current_dir / "test_dataset.json", orient="records")

import sys

from sentence_transformers import SentenceTransformer

from utils import load_datasets, create_matryoshka_evaluator, get_device
from const import MATRYOSHKA_DIMENSIONS, EMBEDDING_MODEL_ID

model_id = EMBEDDING_MODEL_ID
if len(sys.argv) >= 2:
    model_id = sys.argv[1]

fine_tuned_model = SentenceTransformer(
    model_id,
    device=get_device(),
)
# create the evaluator
datasets = load_datasets()
evaluator = create_matryoshka_evaluator(
    test_dataset=datasets.test,
    corpus_dataset=datasets.corpus,
)
results = evaluator(fine_tuned_model)
# print the main score
for dim in MATRYOSHKA_DIMENSIONS:
    key = f"dim_{dim}_cosine_ndcg@10"
    print(f"{key}: {results[key]}")

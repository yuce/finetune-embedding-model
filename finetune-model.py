from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer

from utils import load_datasets, create_matryoshka_evaluator, output_path
from const import EMBEDDING_MODEL_ID, MATRYOSHKA_DIMENSIONS, BATCH_SIZE, TRAIN_EPOCHS

model = SentenceTransformer(
    EMBEDDING_MODEL_ID,
    model_kwargs={"attn_implementation": "sdpa"},
)
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model,
    inner_train_loss,
    matryoshka_dims=MATRYOSHKA_DIMENSIONS,
)
args = SentenceTransformerTrainingArguments(
    output_dir=output_path("fine-tuned-embedding-model_checkpoints"),
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    tf32=True,
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_dim_128_cosine_ndcg@10",
)
datasets = load_datasets()
evaluator = create_matryoshka_evaluator(
    test_dataset=datasets.test,
    corpus_dataset=datasets.corpus,
)
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=datasets.train.select_columns(["anchor", "positive"]),
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()
trainer.save_model(output_path("fine-tuned-embedding-model"))

# set the environment so it is loaded to .cache
# use this for aspect based TAM https://huggingface.co/blog/setfit-absa
import numpy as np
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import precision_recall_fscore_support

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("imdb").shuffle(seed=42)  # shuffle so it's not ordered by label
# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["test"].select(range(100))
test_dataset = dataset["test"].select(range(100, 872))

print(np.unique(train_dataset["label"], return_counts=True))  # 8, 8
print(np.unique(eval_dataset["label"], return_counts=True))  # 53, 47
print(np.unique(test_dataset["label"], return_counts=True))  # 12447, 12453


# Load a SetFit model from Hub
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    labels=["negative", "positive"],
)

args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={
        "text": "text",
        "label": "label",
    },  # Map dataset columns to text/label expected by trainer
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate(test_dataset)
print(metrics)


# Run inference
preds = model.predict(
    ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]
)
print(preds)

preds_test = model.predict(test_dataset["text"])
preds_true = test_dataset["label"]
preds_test = [1 if i == "positive" else 0 for i in preds_test]

prec, rec, f1, _ = precision_recall_fscore_support(
    preds_true, preds_test, average="binary"
)

print({"Precision": prec, "Recall": rec, "F1": f1})

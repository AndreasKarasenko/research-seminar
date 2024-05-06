import numpy as np
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import precision_recall_fscore_support

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("sst2").shuffle(seed=42)  # shuffle so it's not ordered by label

# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["validation"].select(range(100))
test_dataset = dataset["validation"].select(range(100, len(dataset["validation"])))


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
    sampling_strategy="unique",
)
# see https://huggingface.co/docs/setfit/conceptual_guides/sampling_strategies

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={
        "sentence": "text",
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
# ["positive", "negative"]

preds_test = model.predict(test_dataset["sentence"])
preds_true = test_dataset["label"]
preds_test = [1 if i == "positive" else 0 for i in preds_test]

prec, rec, f1, _ = precision_recall_fscore_support(
    preds_true, preds_test, average="binary"
)

print({"Precision": prec, "Recall": rec, "F1": f1})
# {'Precision': 0.8709677419354839, 'Recall': 0.826530612244898, 'F1': 0.8481675392670157}
# Can be even better!

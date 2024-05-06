# Example for scikit-llm a few-shot prompt using chatgpt
# setup your api keys here: https://platform.openai.com/api-keys.
# DO NOT SHARE THE KEY WITH ANYONE
# This key was revoked after the exercise.
from skllm.config import SKLLMConfig

key = "sk-..."
SKLLMConfig.set_openai_key(key)

# set the environment so it is loaded to .cache
# use this for aspect based TAM https://huggingface.co/blog/setfit-absa
import numpy as np
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import precision_recall_fscore_support
from skllm.models.gpt.classification.few_shot import DynamicFewShotGPTClassifier

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("imdb").shuffle(seed=42)  # shuffle so it's not ordered by label
# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=1000)
eval_dataset = dataset["test"].select(range(100))
test_dataset = dataset["test"].select(range(100, 872))

print(np.unique(train_dataset["label"], return_counts=True))  # 8, 8
print(np.unique(eval_dataset["label"], return_counts=True))  # 53, 47
print(np.unique(test_dataset["label"], return_counts=True))  # 12447, 12453

X_train = train_dataset["text"]
y_train = train_dataset["label"]

X_eval = eval_dataset["text"]
y_eval = eval_dataset["label"]

clf = DynamicFewShotGPTClassifier(n_examples=3).fit(
    X_train, y_train
)  # 3 examples per sample and 1000 training samples
# takes about: 4 minutes per index (class)
# costs about: less than 0.2$

# prompt = clf._get_prompt(query)
# print(prompt)
preds = clf.predict(X_eval)
# takes about (start 14:52) end (14:55) 3 minutes
# costs about:

prec, rec, f1, _ = precision_recall_fscore_support(y_eval, preds, average="binary")
# {'Precision': 0.8035714285714286, 'Recall': 0.9574468085106383, 'F1': 0.8737864077669905}

print({"Precision": prec, "Recall": rec, "F1": f1})
# print a confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_eval, preds))
# [[42 11]
#  [ 2 45]]

# Quick recap:
# ~2100 embeddings constructed ~0.06$
# 100 gpt3.5-turbo requests ~0.12$
# total cost: ~0.18$

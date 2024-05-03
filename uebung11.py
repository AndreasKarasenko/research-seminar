#%%
# installiert die pakete in requirements12.txt
# Dieses Script zeigt wie man ein vortrainiertes Model fine tunen kann
# orientiert an
# https://huggingface.co/blog/sentiment-analysis-python#2-how-to-use-pre-trained-sentiment-analysis-models-with-python
# und https://github.com/j-hartmann/siebert/blob/main/SieBERT.ipynb
"""Finetuning bezieht sich hier auf die Anzahl Klassen die wir bestimmen wollen.
Das heißt:
der Kontext ist gleich          -   Kurze Texte zB Reviews, oder Tweets
die Sprache ist gleich          -   zB Englisch
die Aufgabe ist gleich          -   Sentimentanalyse
die Anzahl Klassen ist anders   -   3 anstelle von 2

Beachtet, dass man auf die Modelstruktur achten muss, wie wurde es trainiert, welche Daten erwartet es,
welche Sprache verwendet es, was für eine Art Text wurde verwendet, etc.
"""
# %% importieren der pakete
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding, create_optimizer
from scipy.special import softmax
from datasets import load_dataset, Dataset

# %% aufstellen des models
base = "nlptown/bert-base-multilingual-uncased-sentiment"

# model = TFAutoModelForSequenceClassification.from_pretrained(base)
tokenizer = AutoTokenizer.from_pretrained(base)

# %% Beispielhafte Evaluation
# Auskommentieren wenn man es ausprobieren will
texts = ["This is great", "It could be better", "Worst thing I have ever seen", "It's not perfect, but it's getting there"]
tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

model = TFAutoModelForSequenceClassification.from_pretrained(base)
output = model(**tokens)

predictions = output.logits
scores = softmax(predictions, axis=1)
print(scores)
confidences = scores.max(1).tolist() # gibt die berechnete Klassenwahrscheinlichkeit des Models
classes = scores.argmax(-1).tolist() # gibt das Label zurück
pd.DataFrame(list(zip(texts, classes, confidences)), columns=["text", "class", "confidence"])

# Die Ergebnisse hier sind nicht sonderlich zufriedenstellend. Einige eher neutrale Texte werden als positiv, oder negativ deklariert.
# %% Fine tuning
# Datensatz laden und verfügbare Labels anzeigen
# Der Datensatz kann leider aktuell nicht über Huggingface geladen werden weil der ursprüngliche Link nicht mehr funktioniert
# Aus dem lokalen ZIP laden
import zipfile
import pandas as pd
z = zipfile.ZipFile("./trainingandtestdata.zip")
names = z.namelist() # die Dateien im zipfile
print(names)
col_names = ["sentiment", "tweet_id", "date", "query", "user_name", "message"]
files = [pd.read_csv(z.open(i),encoding="ISO-8859-1",
                     delimiter=",", quotechar='"', header=None, names=col_names) for i in names]
sentiment = files[0]
sentiment.rename(columns={"message":"text", "sentiment": "label"}, inplace=True) # umbenennen der Spalte message zu text
sentiment = Dataset.from_pandas(sentiment)

labels = len(set(sentiment["label"]))
print(labels) # anzahl an labels
print(sentiment) # struktur des Datensatzes

#%%
# Tokenizing
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_sentiment = sentiment.map(preprocess, batched=True)

NUM_EPOCHS = 5
NUM_EXAMPLES = 400

# Dynamischen Padding
# %%
# labels von 0 2 4 zu 0 1 2 transformieren
def update_labels(example):
    example["label"] = example["label"] / 2
    return example

tokenized_sentiment = tokenized_sentiment.map(update_labels)

# %% train und val data bestimmen und das model entsprechend aufsetzen
train_data = tokenized_sentiment.select(range(0, NUM_EXAMPLES))
val_data = tokenized_sentiment.select(range(NUM_EXAMPLES, 498))

batch_size = 4
num_epochs = NUM_EPOCHS
batches_per_epoch = len(train_data) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

model = TFAutoModelForSequenceClassification.from_pretrained(base,
                                                             num_labels=labels, ignore_mismatched_sizes=True)
model.compile(optimizer=optimizer)

# Wir können Datensätze auch zu Tensorflow Datensätzen transformieren
# Hierfür geben wir an welche Spalten wir behalten wollen (WICHTIG die drei gezeigten sind notwendig)
# ob wir zufällige Reihenfolgen wollen
# wie groß die batch size sein soll (gleichzeitig evaluierte Daten)
# und welche Funktion zum vorverarbeiten genommen werden soll
train = train_data.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=4,
    collate_fn=collator
)

test = val_data.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=4,
    collate_fn=collator
)
# %% trainieren
model.fit(train, epochs=2) # takes around 45s - 1m on a 3060

# %% predictions
predictions = model.predict(test)

# %% 
logits = predictions.logits
scores = softmax(logits, axis=1)
confidences = scores.max(1).tolist() # gibt die berechnete Klassenwahrscheinlichkeit des Models
classes = scores.argmax(-1).tolist() # gibt das Label zurück
# %%
results = pd.DataFrame(list(zip(val_data["text"], val_data["label"], classes, confidences)),
                       columns=["text","true","predicted","confidence"])
# %%
results.head()

# %% statistiken berechnen
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

precision, recall, f1, _ = precision_recall_fscore_support(results["true"], results["predicted"],
                                                           average="macro")
acc = accuracy_score(results["true"], results["predicted"])
# %%
print({"accuracy":acc,
       "precision": precision,
       "recall": recall,
       "f1": f1})
# %%
# nach nur einer epoche mit wenig trainingsdaten!
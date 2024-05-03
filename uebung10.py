# %%
# installiert die pakete in requirements12.txt
# Dieses Script zeigt wie man ein vortrainiertes Model verwenden kann
# orientiert an
# https://huggingface.co/blog/sentiment-analysis-python#2-how-to-use-pre-trained-sentiment-analysis-models-with-python
# %% importieren der pakete
from datasets import Dataset, load_dataset
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TFAutoModelForSequenceClassification,
    create_optimizer,
)

# %% datensatz laden
imdb = load_dataset("imdb")

# %% datensatz-struktur betrachten
print(imdb)

# %% beispiel ausgeben
print(imdb["test"][0])

# %% huggingface dataset in pandas dataframe transformieren
test_data = imdb["test"].to_pandas()
# deskriptive statistik
print(test_data.columns)  # welche spalten hat es
print(test_data["label"].unique())  # welche einzigartigen werte hat es
print(test_data["label"].value_counts())  # wie sieht die verteilung aus
print(
    test_data["text"].apply(lambda x: len(x)).describe()
)  # wie sieht die verteilung der textlänge aus

# kurze erkärung zu lambda funktionen:
"""Lambda Funktionen sind sogenannte "inline functions", d.h. sie existieren nur in dem Kontext der Zeile
und können später nicht mehr verwendet werden. das x im Lambda bezieht sich auf jede einzelne Reihe im DataFrame.
Da wir eine einzelne Spalte übergeben ist es entsprechend eine einzelne Zelle, also ein Review.
Die Operation heißt dann: berechne die Länge von jedem Kommentar und berechne anschließend die Statistik
von allen. Beachtet, dass das describe() außerhalb von apply steht!"""

# %% Vortrainiertes Model laden
# eins der folgenden auswählen, wichtig ist, dass man schaut in welcher Reichweite ein Model die Classification macht
# base = "bhadresh-savani/distilbert-base-uncased-emotion"
# base = "nlptown/bert-base-multilingual-uncased-sentiment"
# base = "cardiffnlp/twitter-roberta-base-sentiment"
base = "siebert/sentiment-roberta-large-english"

# Tokenizer bereiten den Text so vor, dass Sie von LLMs konsumiert werden können
# Wörter werden in sub-wörter unterteilt, special tokens für End-of-Line / START / END etc. werden eingefügt
# Encoding und Decoding wird komplett übernommen.
# Es sollte darauf geachtet werden, dass der Tokenizer und das model übereinstimmen (bei beiden base verwenden!)
# Vgl auch https://huggingface.co/docs/transformers/main_classes/tokenizer
tokenizer = AutoTokenizer.from_pretrained(base)


def preprocess_function(examples):
    return tokenizer(
        examples["text"], truncation=True, return_tensors="tf", padding=True
    )


tokenized_imdb = imdb.map(preprocess_function, batched=True)

# %% model definieren
model = TFAutoModelForSequenceClassification.from_pretrained(base)
# batch size, epochs und learning rate sind nur wichtig wenn man es weiter trainieren will.
# damit wir das model aber compilen können und später model.predict auf dem imdb Datensatz verwenden können
batch_size = 4
num_epochs = 3
batches_per_epoch = len(imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(
    init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps
)
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

model.compile(optimizer=optimizer)
# %% testoutput
text = "Good night 😊"
encoded_input = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
output = model(encoded_input)
scores = output.logits.numpy()
scores = softmax(scores)
print(scores)
# scores sind für negativ, neutral, positiv bei cardiffnlp!
# scores sind für negativ, positiv bei siebert!

# %% testoutput für mehrere
text = ["Good night 😊", "Bad night :(", "a regular night"]
encoded_input = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
output = model(encoded_input)
scores = output.logits.numpy()
scores = softmax(scores, axis=1)  # WICHTIG sonst wird spaltenweise softmax berechnet!!!
print(scores)
# %% output für imdb
# alternative um etwas daten zu bekommen, garantiert nicht, dass alle vertreten sind!!!
# tokenized_test = tokenized_imdb["test"].select(range(0,500))
# wir nehmen nur einen kleinen Teil weil es sonst zu lange dauert!
tokenized_test = tokenized_imdb["test"].to_pandas()
sample_indices = tokenized_test["label"].sample(n=500).index
tokenized_test = tokenized_test[tokenized_test.index.isin(sample_indices)]
tokenized_test = Dataset.from_pandas(tokenized_test)
test = tokenized_test.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=8,
    collate_fn=collator,
)
output = model.predict(test)
scores = output.logits
scores = softmax(scores, axis=1)  # WICHTIG sonst wird spaltenweise softmax berechnet!!!
print(scores)
# %%
import pandas as pd

confidences = scores.max(
    1
).tolist()  # gibt die berechnete Klassenwahrscheinlichkeit des Models
classes = scores.argmax(-1).tolist()  # gibt das Label zurück
true_label = tokenized_test["label"]
texts = tokenized_test["text"]

results = pd.DataFrame(
    list(zip(texts, true_label, classes, confidences)),
    columns=["text", "true", "predicted", "confidence"],
)

from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1, _ = precision_recall_fscore_support(
    results["true"], results["predicted"], average="macro"
)
# ACHTUNG wenn nur eine Klasse vertreten ist macro rausnehmen!
print(prec)
print(rec)
print(f1)

# Das sieht schon mal SEHR gut aus
# Wenn wir aber 3 oder mehr klassen bestimmen wollen müssen wir etwas ändern

# %%
# alternativ mit einer pipeline
from transformers import pipeline

base = "cardiffnlp/twitter-roberta-base-sentiment"
pipe = pipeline(
    "sentiment-analysis", base, device=0
)  # funktioniert nur wenn ihr eine GPU mit CUDA support habt!

# %% trainings zyklus definieren
results = []

batch_size = 32  # immer dran denken mit der batch_size zu spielen und zu schauen wie viel euer PC schafft
for i in range(0, len(imdb["test"]), batch_size):
    batch = imdb["test"]["text"][i : i + batch_size]
    output = pipe(batch)
    results.extend(output)
    print(i)  # uncomment if needed

# %%
results

# %%

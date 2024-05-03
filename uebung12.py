#%%
# vgl https://huggingface.co/docs/transformers/v4.17.0/en/tasks/sequence_classification
# installiert die pakete in requirements10.txt
# In diesem Script wird gezeigt wie man transfer learning verwenden kann um ein allgemeines Model auf einen
# spezifischen Kontext zu trainieren
"""Anders als in Übung 11 wollen wir jetzt den Kontext ändern.
das heißt:
die Art der Daten bleibt gleich     -       Unstrukturierte Daten / Text
die Aufgabe hat sich verändert      -       Sentimentanalyse anstelle von Next-Word-Prediction o.ä."""
# %% importieren der pakete
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification

#%% datensatz laden
imdb = load_dataset("imdb")

#%% datensatz-struktur betrachten
print(imdb)

#%% beispiel ausgeben
print(imdb["test"][0])

#%% Preprocessing
# an der Stelle kann man auch andere Modelle verwenden! Sucht dafür auf der Huggingface Seite
base = "distilbert-base-uncased" 

# Tokenizer bereiten den Text so vor, dass Sie von LLMs konsumiert werden können
# Wörter werden in sub-wörter unterteilt, special tokens für End-of-Line / START / END etc. werden eingefügt
# Encoding und Decoding wird komplett übernommen. 
# Es sollte darauf geachtet werden, dass der Tokenizer und das model übereinstimmen (bei beiden base verwenden!) 
# Vgl auch https://huggingface.co/docs/transformers/main_classes/tokenizer
tokenizer = AutoTokenizer.from_pretrained(base)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
#%% metriken definieren
# Vergleiche https://huggingface.co/evaluate-metric
# zB für F1 https://huggingface.co/spaces/evaluate-metric/f1
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

#%% label mappings erstellen
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

#%% trainings-setup
from transformers import create_optimizer
import tensorflow as tf

batch_size = 4
num_epochs = 3
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

#%% model definieren

model = TFAutoModelForSequenceClassification.from_pretrained(base, num_labels=2, id2label=id2label,
                                             label2id=label2id)
#%% erstellen des trainingsdatensatzes
tf_train_set = model.prepare_tf_dataset(
    tokenized_imdb["train"],
    shuffle=True,
    batch_size=4,
    collate_fn=data_collator
)
#%% erstellen des testdatensatzes
tf_test_set = model.prepare_tf_dataset(
    tokenized_imdb["test"],
    shuffle=False,
    batch_size=4,
    collate_fn=data_collator
)
#%% model kompilieren
import tensorflow as tf
model.compile(optimizer=optimizer)


#%% callbacks definieren
from transformers.keras_callbacks import KerasMetricCallback
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)
# %% callback liste definieren
# (hier könnten noch weitere callbacks definiert werden und der liste beigefügt werden)
callbacks = [metric_callback]
# %% model trainieren
model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=1, callbacks=callbacks)
# training dauert selbst mit GPU SEHR lange! Ca 20-30 Minuten je Epoche
# was für alternativen gibt es? Pre-Trained model verwenden! Und lediglich die Inferenz-Zeit in Kauf nehmen.

# %% bestimmen von text-sentimenten
text = "This was pretty easy! I thought this would be a lot harder to be quite honest!"
inputs = tokenizer(text, return_tensors="tf")
outputs = model(**inputs).logits

# %% klasse bestimmen
predicted_class_id = int(tf.math.argmax(outputs, axis=-1)[0])
print(model.config.id2label[predicted_class_id])
# %%

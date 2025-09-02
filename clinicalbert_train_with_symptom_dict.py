import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Sanity check
print("Running the correct script!")
print("TrainingArguments from:", TrainingArguments.__module__)

# Load data
df_pos = pd.read_csv("clean_pos.csv")
df_pos["label"] = 1

df_neg = pd.read_csv("clean_neg.csv")
df_neg["label"] = 0

df = pd.concat([df_pos, df_neg], ignore_index=True)
df = df[["TEXT", "label"]].dropna()
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def tokenize_function(example):
    return tokenizer(example["TEXT"], padding="max_length", truncation=True, max_length=512)

dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Train-test split
split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
test_dataset = split["test"]

# Model
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)

# Training arguments
args = TrainingArguments(
    output_dir="bert_output",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=5,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Evaluate
preds_output = trainer.predict(test_dataset)
preds = preds_output.predictions.argmax(axis=1)
labels = preds_output.label_ids

# Classification report
print("\nClassification Report:\n")
print(classification_report(labels, preds))

# Confusion matrix
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

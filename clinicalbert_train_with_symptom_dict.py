import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
import matplotlib.pyplot as plt

# Symptom keyword dictionary (for reference, not used in training)
symptom_dictionary = {
    "fever": ["fever", "febrile", "pyrexia"],
    "hypothermia": ["hypothermia", "low body temp"],
    "tachycardia": ["tachycardia", "high heart rate"],
    "hypotension": ["hypotension", "low blood pressure"],
    "confusion": ["confused", "altered mental", "disoriented"],
    "dyspnoea": ["shortness of breath", "dyspnoea", "difficulty breathing"],
    "infection_keywords": ["infection", "pneumonia", "UTI", "bacteremia", "septicemia", "sepsis"]
}

# Load and label data
pos = pd.read_csv("clean_pos.csv").assign(label=1)
neg = pd.read_csv("clean_neg.csv").assign(label=0)
df = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)
df = df[["text", "label"]].dropna()

# Tokenize
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

dataset = Dataset.from_pandas(df)
tokenized = dataset.map(tokenize_function, batched=True)

# Train-test split
split = tokenized.train_test_split(test_size=0.2)
train, test = split["train"], split["test"]

# Model
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)

# Training arguments
args = TrainingArguments(
    output_dir="bert_output",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
)

trainer.train()

# Evaluate
preds = trainer.predict(test)
y_pred = preds.predictions.argmax(axis=-1)
y_true = preds.label_ids

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("ClinicalBERT Confusion Matrix")
plt.savefig("confusion_matrix_bert.png")
plt.close()
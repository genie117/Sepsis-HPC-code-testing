import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
import matplotlib.pyplot as plt

# Step 1: Define ASP symptom dictionary
symptom_dictionary = {
    "fever": ["fever", "febrile", "pyrexia"],
    "hypothermia": ["hypothermia", "low body temp", "cold to touch"],
    "tachycardia": ["tachycardia", "high heart rate", "hr>100", "hr > 100"],
    "hypotension": ["hypotension", "low blood pressure", "bp<90", "bp < 90"],
    "confusion": ["confused", "altered mental", "disoriented", "delirium", "gcs", "confusion"],
    "dyspnoea": ["shortness of breath", "dyspnoea", "difficulty breathing", "increased rr", "tachypnoea"],
    "infection": ["infection", "pneumonia", "uti", "bacteremia", "septicemia", "infected wound", "sepsis"]
}

# Step 2: Load datasets
df_pos = pd.read_csv("clean_pos.csv")
df_pos["label"] = 1
df_neg = pd.read_csv("clean_neg.csv")
df_neg["label"] = 0

# Step 3: ASP keyword filtering function
def contains_asp_symptom(text):
    text_lower = str(text).lower()
    for keywords in symptom_dictionary.values():
        if any(keyword in text_lower for keyword in keywords):
            return True
    return False

# Step 4: Filter based on ASP keywords
df_pos["asp_flagged"] = df_pos["TEXT"].apply(contains_asp_symptom)
df_neg["asp_flagged"] = df_neg["TEXT"].apply(contains_asp_symptom)

df_pos = df_pos[df_pos["asp_flagged"]]
df_neg = df_neg[df_neg["asp_flagged"]]

# Step 5: Limit to 500 positive and 500 negative samples for quick testing
df_pos = df_pos.head(500)
df_neg = df_neg.head(500)

# Step 6: Combine, clean, and shuffle
df = pd.concat([df_pos, df_neg], ignore_index=True)
df = df[["TEXT", "label"]].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total ASP-filtered samples used for training/testing: {len(df)}")

# Step 7: Tokenization
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def tokenize_function(example):
    return tokenizer(example["TEXT"], truncation=True, padding="max_length", max_length=512)

dataset = Dataset.from_pandas(df)
tokenized = dataset.map(tokenize_function, batched=True)

# Step 8: Train-test split
split = tokenized.train_test_split(test_size=0.2, seed=42)
train, test = split["train"], split["test"]

# Step 9: Load ClinicalBERT model
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)

# Step 10: Training arguments (fixed save/eval strategies)
args = TrainingArguments(
    output_dir="bert_output_asp",
    evaluation_strategy="epoch",   # evaluate every epoch
    save_strategy="epoch",         # save every epoch (required for load_best_model_at_end)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Step 11: Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
)

# Step 12: Train model
trainer.train()

# Step 13: Evaluate model
preds = trainer.predict(test)
y_pred = preds.predictions.argmax(axis=-1)
y_true = preds.label_ids

# Step 14: Confusion matrix and report
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("ClinicalBERT Confusion Matrix (ASP-filtered)")
plt.savefig("confusion_matrix_bert_asp.png")
plt.close()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# === Step 15: Inspect sample predictions ===
import pandas as pd

# Convert test dataset back into a DataFrame
df_test = pd.DataFrame(test)  # HuggingFace Dataset -> DataFrame
df_test["true_label"] = y_true
df_test["pred_label"] = y_pred

# Map to human-readable labels
label_map = {0: "Non-Sepsis", 1: "Sepsis"}
df_test["true_label"] = df_test["true_label"].map(label_map)
df_test["pred_label"] = df_test["pred_label"].map(label_map)

# Select 5 correct and 5 incorrect samples
correct = df_test[df_test["true_label"] == df_test["pred_label"]].sample(5, random_state=42)
incorrect = df_test[df_test["true_label"] != df_test["pred_label"]].sample(5, random_state=42)

# Combine them
samples = pd.concat([correct, incorrect])

# Save to CSV
samples.to_csv("sample_predictions.csv", index=False)

print("\n Saved 10 sample predictions (5 correct, 5 incorrect) to 'sample_predictions.csv'")

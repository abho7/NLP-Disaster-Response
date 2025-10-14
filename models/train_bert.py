"""
models/train_bert.py

Minimal, reproducible BERT fine-tuning script for binary urgency classification.
Assumes data CSVs at data/train.csv and data/test.csv with columns: text,label
Label: 0 (not urgent), 1 (urgent) — adapt as needed.
"""

from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, ClassLabel, DatasetDict
import numpy as np
import os
import torch

# Configuration (match your paper)
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5
OUTPUT_DIR = "./results/bert-checkpoint"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Load CSV dataset (expect train.csv and test.csv in data/)
data_files = {}
if os.path.exists("data/train.csv"):
    data_files["train"] = "data/train.csv"
if os.path.exists("data/test.csv"):
    data_files["test"] = "data/test.csv"
if os.path.exists("data/validation.csv"):
    data_files["validation"] = "data/validation.csv"

if len(data_files) == 0:
    raise FileNotFoundError("Place train.csv and test.csv in the data/ folder or adapt the script.")

dataset = load_dataset("csv", data_files=data_files)

# Ensure label column is int and in expected range
def preprocess_examples(examples):
    # rename if needed: assume column 'text' and 'label'
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

dataset = dataset.map(preprocess_examples, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=3,
    seed=42,
)

# Simple metrics function
import sklearn.metrics as metrics

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = (preds == labels).mean()
    report = metrics.classification_report(labels, preds, output_dict=True)
    try:
        # compute AUC on predicted probabilities of class 1
        probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()
        fpr, tpr, _ = metrics.roc_curve(labels, probs)
        roc_auc = metrics.auc(fpr, tpr)
    except Exception:
        roc_auc = 0.0
    return {
        "accuracy": accuracy,
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "roc_auc": roc_auc,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset.get("test", None),
    compute_metrics=compute_metrics,
)

# Train
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Training finished. Model saved to", OUTPUT_DIR)

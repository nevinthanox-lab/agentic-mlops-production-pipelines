"""
PromptShield - Phase 3, Step 3.2
Fine-tunes a DistilBERT sequence classification head on the 5-category
prompt dataset (benign, direct_injection, roleplay, obfuscation,
instruction_override) using PyTorch + Hugging Face Trainer + Accelerate.
Logs training metrics to Weights & Biases for the bake-off comparison
against the sklearn baseline.

Reads:  data/processed/full_dataset.csv
Writes:
  models/distilbert_promptshield/          (final fine-tuned model + tokenizer)
  reports/distilbert_confusion_matrix.png
  reports/distilbert_calibration_curve.png
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    classification_report,
)
from loguru import logger
import wandb

logger.add("logs/distilbert_training.log", rotation="5 MB")

PROCESSED_DIR = "data/processed"
MODEL_OUTPUT_DIR = "models/distilbert_promptshield"
REPORTS_DIR = "reports"
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


def load_and_tokenize():
    logger.info("Loading full_dataset.csv ...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "full_dataset.csv"))
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["category"])

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL_NAME)

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=MAX_LENGTH, padding=False
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)

    return train_ds, val_ds, test_ds, test_df, tokenizer, label_encoder


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    }


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_calibration_curve(y_true_binary, y_prob, save_path):
    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="DistilBERT")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability (attack)")
    plt.ylabel("Fraction of positives (actual attack rate)")
    plt.title("Calibration Curve - DistilBERT - Benign vs Attack")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved calibration curve to {save_path}")


def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    wandb.init(project="promptshield", name="distilbert-finetune", job_type="training")

    train_ds, val_ds, test_ds, test_df, tokenizer, label_encoder = load_and_tokenize()
    num_labels = len(label_encoder.classes_)

    logger.info(f"Loading base model {BASE_MODEL_NAME} with {num_labels} labels ...")
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=num_labels
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="models/distilbert_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir="logs/distilbert_tb",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        report_to=["wandb"],
        save_total_limit=2,
        run_name="distilbert-finetune",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting DistilBERT fine-tuning ...")
    trainer.train()

    logger.info("Evaluating on held-out test set ...")
    test_predictions = trainer.predict(test_ds)
    logits = test_predictions.predictions
    y_pred = np.argmax(logits, axis=1)
    y_true = test_predictions.label_ids

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_labels)), zero_division=0
    )

    report_text = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print("\n--- DistilBERT Classification Report ---")
    print(report_text)

    plot_confusion_matrix(
        y_true, y_pred, label_encoder.classes_,
        "DistilBERT - Confusion Matrix",
        os.path.join(REPORTS_DIR, "distilbert_confusion_matrix.png"),
    )

    # Calibration curve: benign=0 vs any-attack=1, using softmax probabilities
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    benign_idx = list(label_encoder.classes_).index("benign")
    y_true_binary = (y_true != benign_idx).astype(int)
    attack_prob = 1.0 - probs[:, benign_idx]
    plot_calibration_curve(
        y_true_binary, attack_prob,
        os.path.join(REPORTS_DIR, "distilbert_calibration_curve.png"),
    )

    logger.info(f"DistilBERT test accuracy: {accuracy:.4f}")

    wandb.log({
        "distilbert_test_accuracy": accuracy,
        "distilbert_confusion_matrix": wandb.Image(os.path.join(REPORTS_DIR, "distilbert_confusion_matrix.png")),
        "distilbert_calibration_curve": wandb.Image(os.path.join(REPORTS_DIR, "distilbert_calibration_curve.png")),
    })

    logger.info(f"Saving final model + tokenizer to {MODEL_OUTPUT_DIR} ...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    # Save label encoder classes alongside the model for inference-time decoding
    with open(os.path.join(MODEL_OUTPUT_DIR, "label_classes.txt"), "w") as f:
        for cls in label_encoder.classes_:
            f.write(f"{cls}\n")

    print("\n=== DISTILBERT FINE-TUNING DONE ===")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}/")

    wandb.finish()


if __name__ == "__main__":
    main()

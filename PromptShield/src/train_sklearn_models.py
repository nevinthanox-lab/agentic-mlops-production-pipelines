"""
PromptShield - Phase 3, Step 3.1
Trains and compares two scikit-learn classifiers (GradientBoosting and
LogisticRegression) on dense sentence embeddings concatenated with the
5 hand-engineered features. Logs metrics, confusion matrix, and the
calibration curve to Weights & Biases for a bake-off comparison.

Reads:
  data/processed/embeddings.npy
  data/processed/features.csv
  data/processed/full_dataset.csv

Writes:
  models/sklearn_gb_model.joblib
  models/sklearn_logreg_model.joblib
  models/label_encoder.joblib
  reports/sklearn_confusion_matrix_gb.png
  reports/sklearn_confusion_matrix_logreg.png
  reports/sklearn_calibration_curve.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import wandb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    classification_report,
)
from loguru import logger

logger.add("logs/sklearn_training.log", rotation="5 MB")

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def load_data():
    logger.info("Loading embeddings, features, and full dataset ...")
    embeddings = np.load(os.path.join(PROCESSED_DIR, "embeddings.npy"))
    features_df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
    full_df = pd.read_csv(os.path.join(PROCESSED_DIR, "full_dataset.csv"))

    # Align by row_id to guarantee correct row order
    features_df = features_df.set_index("row_id").loc[full_df["row_id"]].reset_index()
    feature_matrix = features_df.drop(columns=["row_id"]).values

    embeddings_aligned = embeddings[full_df["row_id"].values]

    X = np.concatenate([embeddings_aligned, feature_matrix], axis=1)
    y = full_df["category"].values
    split = full_df["split"].values

    logger.info(f"Full feature matrix shape: {X.shape}")
    return X, y, split, full_df


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_calibration_curve(y_true_binary, y_prob, model_name, save_path):
    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability (attack)")
    plt.ylabel("Fraction of positives (actual attack rate)")
    plt.title("Calibration Curve - Benign vs Attack")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved calibration curve to {save_path}")


def evaluate_model(model, X_test, y_test_encoded, label_encoder, model_name):
    y_pred = model.predict(X_test)
    labels = list(range(len(label_encoder.classes_)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_encoded, y_pred, labels=labels, zero_division=0
    )
    accuracy = accuracy_score(y_test_encoded, y_pred)

    per_category_metrics = {}
    for idx, cat in enumerate(label_encoder.classes_):
        per_category_metrics[cat] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    logger.info(f"[{model_name}] Overall accuracy: {accuracy:.4f}")
    logger.info(f"[{model_name}] Per-category metrics: {per_category_metrics}")

    report_text = classification_report(
        y_test_encoded, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print(f"\n--- {model_name} Classification Report ---")
    print(report_text)

    return accuracy, per_category_metrics, y_pred


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    run = wandb.init(project="promptshield", name="sklearn-bakeoff", job_type="training")

    X, y, split, full_df = load_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    X_train, y_train = X[train_mask], y_encoded[train_mask]
    X_val, y_val = X[val_mask], y_encoded[val_mask]
    X_test, y_test = X[test_mask], y_encoded[test_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "sklearn_scaler.joblib"))

    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # -------------------------------------------------------------------
    # Model 1: Gradient Boosting
    # -------------------------------------------------------------------
    logger.info("Training GradientBoostingClassifier ...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    joblib.dump(gb_model, os.path.join(MODELS_DIR, "sklearn_gb_model.joblib"))

    gb_accuracy, gb_metrics, gb_preds = evaluate_model(
        gb_model, X_test_scaled, y_test, label_encoder, "GradientBoosting"
    )
    plot_confusion_matrix(
        y_test, gb_preds, list(range(len(label_encoder.classes_))),
        "GradientBoosting - Confusion Matrix",
        os.path.join(REPORTS_DIR, "sklearn_confusion_matrix_gb.png"),
    )

    # -------------------------------------------------------------------
    # Model 2: Logistic Regression
    # -------------------------------------------------------------------
    logger.info("Training LogisticRegression ...")
    logreg_model = LogisticRegression(
        max_iter=2000, C=1.0, random_state=42
    )
    logreg_model.fit(X_train_scaled, y_train)
    joblib.dump(logreg_model, os.path.join(MODELS_DIR, "sklearn_logreg_model.joblib"))

    logreg_accuracy, logreg_metrics, logreg_preds = evaluate_model(
        logreg_model, X_test_scaled, y_test, label_encoder, "LogisticRegression"
    )
    plot_confusion_matrix(
        y_test, logreg_preds, list(range(len(label_encoder.classes_))),
        "LogisticRegression - Confusion Matrix",
        os.path.join(REPORTS_DIR, "sklearn_confusion_matrix_logreg.png"),
    )

    # -------------------------------------------------------------------
    # Calibration curve (binary: benign=0 vs any-attack=1) using GB probabilities
    # -------------------------------------------------------------------
    benign_idx = list(label_encoder.classes_).index("benign")
    y_test_binary = (y_test != benign_idx).astype(int)
    gb_probs = gb_model.predict_proba(X_test_scaled)
    gb_attack_prob = 1.0 - gb_probs[:, benign_idx]
    plot_calibration_curve(
        y_test_binary, gb_attack_prob, "GradientBoosting",
        os.path.join(REPORTS_DIR, "sklearn_calibration_curve.png"),
    )

    # -------------------------------------------------------------------
    # Log everything to W&B
    # -------------------------------------------------------------------
    wandb.log({
        "gb_test_accuracy": gb_accuracy,
        "logreg_test_accuracy": logreg_accuracy,
    })
    for cat, metrics in gb_metrics.items():
        wandb.log({f"gb_{cat}_precision": metrics["precision"], f"gb_{cat}_recall": metrics["recall"], f"gb_{cat}_f1": metrics["f1"]})
    for cat, metrics in logreg_metrics.items():
        wandb.log({f"logreg_{cat}_precision": metrics["precision"], f"logreg_{cat}_recall": metrics["recall"], f"logreg_{cat}_f1": metrics["f1"]})

    wandb.log({
        "gb_confusion_matrix": wandb.Image(os.path.join(REPORTS_DIR, "sklearn_confusion_matrix_gb.png")),
        "logreg_confusion_matrix": wandb.Image(os.path.join(REPORTS_DIR, "sklearn_confusion_matrix_logreg.png")),
        "calibration_curve": wandb.Image(os.path.join(REPORTS_DIR, "sklearn_calibration_curve.png")),
    })

    print("\n=== SKLEARN BAKE-OFF DONE ===")
    print(f"GradientBoosting test accuracy: {gb_accuracy:.4f}")
    print(f"LogisticRegression test accuracy: {logreg_accuracy:.4f}")
    print(f"Models saved to: {MODELS_DIR}/")
    print(f"Reports saved to: {REPORTS_DIR}/")

    wandb.finish()


if __name__ == "__main__":
    main()

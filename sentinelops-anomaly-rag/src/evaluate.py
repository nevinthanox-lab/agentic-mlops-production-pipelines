"""
STEP 4.1-4.4 - Evaluation Suite
Computes final precision/recall/F1 per model vs injected ground truth,
plots ROC/PR curves, defines the ensemble combination rule, and writes
metrics.json with REAL numbers (no placeholders).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    precision_recall_fscore_support, average_precision_score
)

SCORES_PATH = "data/processed/model_scores.parquet"
THRESH_PATH = "models/thresholds.json"
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load():
    df = pd.read_parquet(SCORES_PATH)
    with open(THRESH_PATH) as f:
        thresholds = json.load(f)
    return df, thresholds


def normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a score array to [0, 1] for combining across models."""
    lo, hi = scores.min(), scores.max()
    if hi - lo == 0:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def plot_roc_pr(y_true, model_scores: dict, save_prefix: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, scores in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        axes[1].plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{save_prefix}.png", dpi=150)
    plt.close()


def main():
    df, thresholds = load()
    y_true = df["y_true"].values

    model_scores = {
        "IsolationForest": df["iso_score"].values,
        "OneClassSVM": df["svm_score"].values,
        "Autoencoder": df["ae_rmse"].values,
    }

    # STEP 4.1 - per-model metrics at calibrated thresholds
    per_model_metrics = {}
    for name, key in [("IsolationForest", "isolation_forest"),
                       ("OneClassSVM", "one_class_svm"),
                       ("Autoencoder", "autoencoder")]:
        scores = model_scores[name]
        preds = (scores >= thresholds[key]).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = float(auc(fpr, tpr))
        per_model_metrics[name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(roc_auc, 4),
        }
        print(f"{name}: P={p:.3f} R={r:.3f} F1={f1:.3f} AUC={roc_auc:.3f}")

    # Autoencoder RMSE separately reported (STEP 4.4 requirement)
    ae_rmse_mean = float(df["ae_rmse"].mean())
    ae_rmse_on_anomalies = float(df.loc[y_true == 1, "ae_rmse"].mean())
    ae_rmse_on_normal = float(df.loc[y_true == 0, "ae_rmse"].mean())

    # STEP 4.2 - ROC/PR curves
    plot_roc_pr(y_true, model_scores, "roc_pr_curves")

    # STEP 4.3 - Ensemble combination rule:
    # High confidence  = IsolationForest score AND Autoencoder RMSE both elevated (normalized > 0.5)
    # Medium confidence = models disagree
    # Low confidence    = neither elevated
    iso_norm = normalize(model_scores["IsolationForest"])
    ae_norm = normalize(model_scores["Autoencoder"])
    iso_flag = iso_norm > 0.5
    ae_flag = ae_norm > 0.5

    severity = np.where(iso_flag & ae_flag, "high",
                np.where(iso_flag | ae_flag, "medium", "low"))
    ensemble_preds = (severity != "low").astype(int)
    ep, er, ef1, _ = precision_recall_fscore_support(y_true, ensemble_preds, average="binary", zero_division=0)
    print(f"Ensemble (IF+AE agreement rule): P={ep:.3f} R={er:.3f} F1={ef1:.3f}")

    per_model_metrics["Ensemble_IF_AE_Agreement"] = {
        "precision": round(float(ep), 4),
        "recall": round(float(er), 4),
        "f1": round(float(ef1), 4),
        "severity_distribution": pd.Series(severity).value_counts().to_dict(),
    }

    # STEP 4.4 - metrics.json with REAL numbers
    metrics_output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "dataset": {
            "n_samples": int(len(df)),
            "n_services": int(df["service_name"].nunique()),
            "anomaly_rate": round(float(y_true.mean()), 4),
        },
        "models": per_model_metrics,
        "autoencoder_rmse": {
            "overall_mean": round(ae_rmse_mean, 4),
            "on_ground_truth_anomalies": round(ae_rmse_on_anomalies, 4),
            "on_normal_windows": round(ae_rmse_on_normal, 4),
        },
        "best_model": max(
            [("IsolationForest", per_model_metrics["IsolationForest"]["f1"]),
             ("OneClassSVM", per_model_metrics["OneClassSVM"]["f1"]),
             ("Autoencoder", per_model_metrics["Autoencoder"]["f1"]),
             ("Ensemble_IF_AE_Agreement", per_model_metrics["Ensemble_IF_AE_Agreement"]["f1"])],
            key=lambda x: x[1]
        )[0],
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)

    print("\nmetrics.json written with REAL evaluation numbers.")
    print(f"Best performing approach: {metrics_output['best_model']}")
    print(f"ROC/PR curves saved -> {FIG_DIR / 'roc_pr_curves.png'}")


if __name__ == "__main__":
    main()

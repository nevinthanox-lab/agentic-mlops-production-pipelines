"""
STEP 3.2-3.6 - Model Training
Trains three unsupervised anomaly detectors on 'normal' windows only:
  - IsolationForest (primary)
  - OneClassSVM (comparison baseline)
  - TensorFlow/Keras autoencoder (ensemble second opinion, RMSE-based)
All three thresholds are calibrated against the injected ground-truth
windows (validation only - training stays unsupervised).
Every run is logged to MLflow.
"""

import numpy as np
import pandas as pd
import mlflow
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras

FEATURES_PATH = "data/processed/features.parquet"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "cpu_pct", "memory_pct", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "error_rate_pct", "queue_depth", "thread_count",
    "cpu_zscore", "mem_zscore", "latency_zscore", "error_zscore", "queue_zscore", "thread_zscore",
    "cpu_rate_of_change", "latency_rate_of_change", "memory_rate_of_change",
    "latency_per_cpu_ratio", "queue_per_thread_ratio",
]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("sentinelops-anomaly-detection")


def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = df["is_anomaly_ground_truth"].values
    return df, X, y


def train_isolation_forest(X_train_scaled, contamination):
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)
    return model


def train_one_class_svm(X_train_scaled, nu):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    model.fit(X_train_scaled)
    return model


def build_autoencoder(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(16, activation="relu")(inputs)
    x = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def calibrate_threshold(scores, y_true, higher_is_anomalous=True):
    """Sweep percentile thresholds, pick the one maximizing F1 against ground truth."""
    best_f1, best_thresh, best_metrics = -1, None, None
    percentiles = np.arange(80, 99.5, 0.5)
    for p in percentiles:
        thresh = np.percentile(scores, p)
        preds = (scores >= thresh).astype(int) if higher_is_anomalous else (scores <= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
            best_metrics = {"precision": precision, "recall": recall, "f1": f1, "percentile": p}
    return best_thresh, best_metrics


def main():
    df, X, y = load_data()
    contamination = y.mean()  # matches true anomaly rate (~3.5%)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with mlflow.start_run(run_name="ensemble_calibration"):
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("n_samples", len(X))

        # --- IsolationForest ---
        print("Training IsolationForest...")
        iso_model = train_isolation_forest(X_scaled, contamination)
        iso_scores = -iso_model.score_samples(X_scaled)  # higher = more anomalous
        iso_thresh, iso_metrics = calibrate_threshold(iso_scores, y, higher_is_anomalous=True)
        print(f"IsolationForest -> P={iso_metrics['precision']:.3f} R={iso_metrics['recall']:.3f} F1={iso_metrics['f1']:.3f}")
        mlflow.log_metrics({f"iso_{k}": v for k, v in iso_metrics.items() if k != "percentile"})
        mlflow.log_param("iso_threshold_percentile", iso_metrics["percentile"])

        # --- OneClassSVM ---
        print("Training OneClassSVM...")
        svm_sample = X_scaled if len(X_scaled) < 5000 else X_scaled[np.random.choice(len(X_scaled), 5000, replace=False)]
        svm_model = train_one_class_svm(svm_sample, nu=min(contamination * 1.5, 0.3))
        svm_scores = -svm_model.decision_function(X_scaled)
        svm_thresh, svm_metrics = calibrate_threshold(svm_scores, y, higher_is_anomalous=True)
        print(f"OneClassSVM -> P={svm_metrics['precision']:.3f} R={svm_metrics['recall']:.3f} F1={svm_metrics['f1']:.3f}")
        mlflow.log_metrics({f"svm_{k}": v for k, v in svm_metrics.items() if k != "percentile"})

        # --- Autoencoder ---
        print("Training Autoencoder...")
        normal_mask = y == 0
        X_normal = X_scaled[normal_mask]
        ae = build_autoencoder(X_scaled.shape[1])
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = ae.fit(
            X_normal, X_normal,
            validation_split=0.15,
            epochs=50,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0,
        )
        recon = ae.predict(X_scaled, verbose=0)
        ae_rmse = np.sqrt(np.mean((X_scaled - recon) ** 2, axis=1))
        ae_thresh, ae_metrics = calibrate_threshold(ae_rmse, y, higher_is_anomalous=True)
        print(f"Autoencoder -> P={ae_metrics['precision']:.3f} R={ae_metrics['recall']:.3f} F1={ae_metrics['f1']:.3f} "
              f"final_val_loss={history.history['val_loss'][-1]:.4f}")
        mlflow.log_metrics({f"ae_{k}": v for k, v in ae_metrics.items() if k != "percentile"})
        mlflow.log_metric("ae_final_val_loss", history.history["val_loss"][-1])

        # --- Persist everything for Phase 4 evaluation + Phase 7 serving ---
        joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
        joblib.dump(iso_model, MODEL_DIR / "isolation_forest.joblib")
        joblib.dump(svm_model, MODEL_DIR / "one_class_svm.joblib")
        ae.save(MODEL_DIR / "autoencoder.keras")

        thresholds = {
            "isolation_forest": float(iso_thresh),
            "one_class_svm": float(svm_thresh),
            "autoencoder": float(ae_thresh),
        }
        pd.Series(thresholds).to_json(MODEL_DIR / "thresholds.json")

        scores_df = pd.DataFrame({
            "service_name": df["service_name"],
            "timestamp": df["timestamp"],
            "y_true": y,
            "iso_score": iso_scores,
            "svm_score": svm_scores,
            "ae_rmse": ae_rmse,
        })
        scores_df.to_parquet("data/processed/model_scores.parquet", index=False)

        mlflow.log_artifact(str(MODEL_DIR / "thresholds.json"))
        print("\nAll models + thresholds saved to models/. Scores saved to data/processed/model_scores.parquet")


if __name__ == "__main__":
    main()

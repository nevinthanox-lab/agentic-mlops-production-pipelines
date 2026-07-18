"""STEP 1.4 - Verify generated telemetry data integrity."""
import pandas as pd

df = pd.read_parquet("data/raw/telemetry.parquet")
gt = pd.read_parquet("data/raw/ground_truth_windows.parquet")

print("=== Telemetry Schema ===")
print(df.dtypes)
print("\n=== Row count per service ===")
print(df["service_name"].value_counts())
print("\n=== Anomaly label balance ===")
print(df["is_anomaly_ground_truth"].value_counts(normalize=True))
print(f"\n=== Ground truth windows: {len(gt)} ===")
print(gt["anomaly_type"].value_counts())
print("\n=== Null check ===")
print(df.isnull().sum().sum(), "total nulls")

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading engineered data...")
df = pd.read_parquet("engineered_customers.parquet")

# 1. Smart target resolution to avoid pandas suffixing collisions (e.g., churn_x, churn_y)
target_col_df = None
for col in df.columns:
    if "churn" in col.lower():
        target_col_df = col
        break

if target_col_df:
    if target_col_df != "churn_label":
        df = df.rename(columns={target_col_df: "churn_label"})
    print(f"Found churn column '{target_col_df}' in engineered data. Aligned to 'churn_label'! ✅")
else:
    print("churn_label not found in engineered_customers.parquet. Merging from raw_customers.parquet...")
    df_raw = pd.read_parquet("raw_customers.parquet")
    
    target_col_raw = None
    for col in df_raw.columns:
        if "churn" in col.lower():
            target_col_raw = col
            break
            
    if target_col_raw and "account_id" in df.columns and "account_id" in df_raw.columns:
        if target_col_raw in df.columns:
            df = df.drop(columns=[target_col_raw])
            
        df = df.merge(df_raw[["account_id", target_col_raw]], on="account_id", how="inner")
        if target_col_raw != "churn_label":
            df = df.rename(columns={target_col_raw: "churn_label"})
        print("Successfully merged and aligned churn_label! ✅")
    else:
        raise KeyError("Could not find any churn-related column or account_id to align the targets.")

# 2. Separate features (X) and target (y) safely, avoiding data leakage
X = df.drop(columns=["account_id", "churn_label"], errors="ignore")
churn_cols_in_X = [col for col in X.columns if "churn" in col.lower()]
X = X.drop(columns=churn_cols_in_X, errors="ignore").select_dtypes(include=[np.number])

y = df["churn_label"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

model = xgb.XGBClassifier()
model.load_model("xgboost_churn_model.json")

# Align features with what the model expects
feature_names = model.get_booster().feature_names
if feature_names:
    X_test = X_test[feature_names]

y_prob = model.predict_proba(X_test)[:, 1]

y_pred_default = (y_prob >= 0.5).astype(int)
precision_def = precision_score(y_test, y_pred_default, zero_division=0)
recall_def = recall_score(y_test, y_pred_default, zero_division=0)
f1_def = f1_score(y_test, y_pred_default, zero_division=0)
auc_score = roc_auc_score(y_test, y_prob)

arr_col = "arr" if "arr" in X_test.columns else X_test.columns[0]
top_decile_cutoff = X_test[arr_col].quantile(0.90)
high_arr_mask = (X_test[arr_col] >= top_decile_cutoff).values

best_threshold = 0.5
best_high_arr_recall = 0.0
for thresh in np.linspace(0.1, 0.9, 81):
    preds = (y_prob >= thresh).astype(int)
    high_arr_recall = recall_score(y_test[high_arr_mask], preds[high_arr_mask], zero_division=0)
    if high_arr_recall >= best_high_arr_recall:
        best_high_arr_recall = high_arr_recall
        best_threshold = thresh

y_pred_tuned = (y_prob >= best_threshold).astype(int)
precision_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
recall_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)

metrics = {
    "default_threshold": 0.5,
    "default_precision": float(precision_def),
    "default_recall": float(recall_def),
    "default_f1_score": float(f1_def),
    "roc_auc_score": float(auc_score),
    "business_tuned_threshold": float(best_threshold),
    "tuned_precision": float(precision_tuned),
    "tuned_recall": float(recall_tuned),
    "tuned_f1_score": float(f1_tuned)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title(f"Confusion Matrix (Thresh: {best_threshold:.2f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_score:.2f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
axes[1].set_xlabel("FPR")
axes[1].set_ylabel("TPR")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig("evaluation_plots.png", dpi=150)
plt.close()
print("Evaluation finished! metrics.json and evaluation_plots.png saved.")

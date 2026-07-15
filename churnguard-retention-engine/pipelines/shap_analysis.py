import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np

print("Loading data and model...")
df = pd.read_parquet("engineered_customers.parquet")

# Safely select features and exclude any leakage targets
X = df.drop(columns=["account_id", "churn_label"], errors="ignore")
churn_cols_in_X = [col for col in X.columns if "churn" in col.lower()]
X = X.drop(columns=churn_cols_in_X, errors="ignore").select_dtypes(include=[np.number])

model = xgb.XGBClassifier()
model.load_model("xgboost_churn_model.json")

# Align features with what the model expects
feature_names = model.get_booster().feature_names
if feature_names:
    print("Aligning features with model input...")
    X = X[feature_names]

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Generating SHAP Summary Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP Summary Plot saved as shap_summary.png! ✅")

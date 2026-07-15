import pandas as pd
import xgboost as xgb
import shap
import json
import numpy as np
from loguru import logger

def run_batch_scoring():
    logger.info("Loading engineered data and trained model...")
    df = pd.read_parquet('engineered_customers.parquet')
    
    # Features and Context
    features = df.drop(columns=['account_id', 'churn'])
    account_ids = df['account_id']
    actual_arr = df['arr']
    
    # Load trained model
    model = xgb.Booster()
    model.load_model('xgboost_churn_model.json')
    dmatrix = xgb.DMatrix(features)
    
    logger.info("Predicting churn probabilities...")
    churn_probs = model.predict(dmatrix)
    
    logger.info("Running SHAP TreeExplainer to extract top drivers...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Filter only High-Risk Accounts (Probability > 0.6)
    THRESHOLD = 0.6
    high_risk_indices = np.where(churn_probs > THRESHOLD)[0]
    
    logger.info(f"Identified {len(high_risk_indices)} high-risk accounts. Building payload...")
    
    feature_names = features.columns.tolist()
    flagged_accounts = []
    
    for idx in high_risk_indices:
        account_id = account_ids.iloc[idx]
        prob = float(churn_probs[idx])
        arr = float(actual_arr.iloc[idx])
        
        # Get local SHAP values for this specific customer
        customer_shap = shap_values[idx]
        
        # Sort by absolute magnitude to find the most impactful features
        top_indices = np.argsort(np.abs(customer_shap))[-3:][::-1]
        
        top_drivers = []
        for i in top_indices:
            driver_name = feature_names[i]
            contribution = float(customer_shap[i])
            # Only include factors pushing TOWARDS churn (positive SHAP value)
            if contribution > 0:
                top_drivers.append(driver_name)
                
        if not top_drivers:
            top_drivers = ["overall_usage_pattern"]
            
        flagged_accounts.append({
            "account_id": str(account_id),
            "churn_probability": round(prob, 3),
            "arr": round(arr, 2),
            "top_3_shap_drivers": top_drivers
        })
        
    # Save payload to JSON
    output_file = "high_risk_accounts_payload.json"
    with open(output_file, 'w') as f:
        json.dump(flagged_accounts, f, indent=4)
        
    logger.info(f"Batch scoring complete! {len(flagged_accounts)} accounts saved to {output_file}")

if __name__ == "__main__":
    run_batch_scoring()
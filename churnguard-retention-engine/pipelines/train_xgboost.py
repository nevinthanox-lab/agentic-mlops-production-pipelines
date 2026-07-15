import pandas as pd
import xgboost as xgb
import optuna
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from loguru import logger

def train_churn_model():
    logger.info("Loading engineered Parquet data...")
    df = pd.read_parquet('engineered_customers.parquet')
    
    # Features and Target
    X = df.drop(columns=['account_id', 'churn'])
    y = df['churn']

    # Stratified 70/15/15 Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)
    
    # Class imbalance handling via scale_pos_weight
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    def objective(trial):
        with mlflow.start_run(nested=True):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "scale_pos_weight": scale_pos_weight,
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dval, "val")],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            preds = model.predict(dval)
            auc = roc_auc_score(y_val, preds)
            
            mlflow.log_params(params)
            mlflow.log_metric("val_auc", auc)
            
            return auc

    logger.info("Starting Optuna hyperparameter search (20 trials)...")
    mlflow.set_experiment("ChurnGuard_XGBoost")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    logger.info("Training final model with best parameters...")
    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["scale_pos_weight"] = scale_pos_weight
    
    dtrain_final = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    final_model = xgb.train(best_params, dtrain_final, num_boost_round=150)
    
    # Evaluation
    test_preds_prob = final_model.predict(dtest)
    test_preds_label = (test_preds_prob > 0.5).astype(int)
    
    test_auc = roc_auc_score(y_test, test_preds_prob)
    test_f1 = f1_score(y_test, test_preds_label)
    
    logger.info(f"Test AUC: {test_auc:.4f} | Test F1: {test_f1:.4f}")
    
    # Save Model
    model_path = "xgboost_churn_model.json"
    final_model.save_model(model_path)
    logger.info(f"Final model saved to {model_path}")

if __name__ == "__main__":
    train_churn_model()
import numpy as np
import pandas as pd
import duckdb
from loguru import logger

# Set random seed for reproducibility
np.random.seed(42)

NUM_ACCOUNTS = 10000

def generate_synthetic_data(file_path: str = "raw_customers.parquet") -> None:
    logger.info(f"Generating {NUM_ACCOUNTS} rows of synthetic SaaS customer data...")

    account_ids = [f"ACC-{i:05d}" for i in range(NUM_ACCOUNTS)]

    # Telemetry & Commercial data
    weekly_active_sessions = np.random.poisson(lam=5, size=NUM_ACCOUNTS)
    feature_adoption_counts = np.random.randint(1, 20, size=NUM_ACCOUNTS)
    seat_utilization_pct = np.random.uniform(0.1, 1.0, size=NUM_ACCOUNTS)

    arr = np.random.uniform(5000, 150000, size=NUM_ACCOUNTS)
    contract_length_months = np.random.choice([12, 24, 36], size=NUM_ACCOUNTS)
    discount_pct = np.random.uniform(0.0, 0.3, size=NUM_ACCOUNTS)

    # Support signals
    ticket_count = np.random.poisson(lam=2, size=NUM_ACCOUNTS)
    csat = np.random.uniform(1.0, 5.0, size=NUM_ACCOUNTS)
    nps = np.random.randint(-100, 100, size=NUM_ACCOUNTS)

    # Non-linear churn label generation via hand-tuned logistic function with Gaussian noise
    z = (
        - 2.5
        - 1.5 * seat_utilization_pct
        + 0.5 * ticket_count
        - 0.8 * (csat - 3.0)
        + np.random.normal(0, 1.0, NUM_ACCOUNTS)
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churn_label = (churn_prob > 0.5).astype(int)

    df = pd.DataFrame({
        "account_id": account_ids,
        "weekly_active_sessions": weekly_active_sessions,
        "feature_adoption_counts": feature_adoption_counts,
        "seat_utilization_pct": seat_utilization_pct,
        "arr": arr,
        "contract_length_months": contract_length_months,
        "discount_pct": discount_pct,
        "ticket_count": ticket_count,
        "csat": csat,
        "nps": nps,
        "churn": churn_label
    })

    # Save to Parquet
    df.to_parquet(file_path)
    logger.info(f"Raw synthetic data saved to {file_path}")

def run_duckdb_feature_engineering(input_file: str = "raw_customers.parquet", output_file: str = "engineered_customers.parquet") -> None:
    logger.info("Running DuckDB feature engineering (RFM & Aggregations)...")

    # DuckDB in-memory connection directly over Parquet file
    con = duckdb.connect(database=':memory:')

    query = f"""
        CREATE TABLE engineered_features AS
        SELECT
            account_id,
            weekly_active_sessions,
            feature_adoption_counts,
            seat_utilization_pct,
            arr,
            contract_length_months,
            ticket_count,
            csat,
            churn,
            -- RFM-style Monetary buckets
            NTILE(4) OVER (ORDER BY arr DESC) AS arr_quartile,
            -- Categorical encoding logic
            CASE WHEN discount_pct > 0.2 THEN 1 ELSE 0 END AS high_discount_flag,
            CASE WHEN seat_utilization_pct < 0.5 THEN 1 ELSE 0 END AS low_utilization_flag,
            -- Mocking a rolling usage delta for this static dataset
            weekly_active_sessions * 4 AS estimated_monthly_sessions
        FROM '{input_file}';
    """
    con.execute(query)

    export_query = f"COPY engineered_features TO '{output_file}' (FORMAT PARQUET);"
    con.execute(export_query)

    logger.info(f"Engineered data saved to {output_file}")
    con.close()

if __name__ == "__main__":
    generate_synthetic_data()
    run_duckdb_feature_engineering()
    logger.info("Data pipeline completed successfully! Ready for XGBoost.")
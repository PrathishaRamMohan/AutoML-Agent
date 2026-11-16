"""Generate a realistic synthetic customer-churn dataset for demos.

It deliberately contains problems the agents should catch:
  * customer_id       -> ID-like column (should be dropped)
  * missing values    -> in monthly_charges, tenure_months, support_calls
  * class imbalance   -> ~20% churners
  * account_closed    -> a LEAKAGE column (almost identical to the target)
  * region_code       -> constant column (useless)

Run:  python scripts/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 3000

tenure = rng.integers(1, 72, n).astype(float)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet = rng.choice(["DSL", "Fiber optic", "No internet"], n, p=[0.35, 0.45, 0.20])
payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
monthly = np.round(rng.normal(65, 25, n).clip(18, 120), 2)
support_calls = rng.poisson(1.5, n).astype(float)
senior = rng.choice([0, 1], n, p=[0.84, 0.16])
data_gb = np.round(rng.gamma(2.0, 10.0, n), 1)

# True churn logic (hidden from the model)
logit = (
    -1.2
    - 0.045 * tenure
    + 1.1 * (contract == "Month-to-month")
    - 0.8 * (contract == "Two year")
    + 0.6 * (internet == "Fiber optic")
    + 0.5 * (payment == "Electronic check")
    + 0.018 * (monthly - 65)
    + 0.35 * support_calls
    + 0.3 * senior
)
p = 1 / (1 + np.exp(-logit))
churn = (rng.random(n) < p).astype(int)

df = pd.DataFrame(
    {
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "monthly_charges": monthly,
        "support_calls": support_calls,
        "senior_citizen": senior,
        "data_usage_gb": data_gb,
        "region_code": "TN-01",
        # leakage: recorded AFTER the customer churned
        "account_closed": np.where(rng.random(n) < 0.98, churn, 1 - churn),
        "churn": churn,
    }
)

# inject missing values
for col, frac in [("monthly_charges", 0.04), ("tenure_months", 0.03), ("support_calls", 0.05)]:
    idx = rng.choice(n, int(n * frac), replace=False)
    df.loc[idx, col] = np.nan

out = Path(__file__).resolve().parent.parent / "data" / "sample_churn.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  shape={df.shape}  churn_rate={churn.mean():.2%}")

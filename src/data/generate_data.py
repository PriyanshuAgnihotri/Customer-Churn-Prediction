import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_data(n_users=10000):
    data = pd.DataFrame({
        "user_id": range(1, n_users+1),
        "signup_date": [datetime(2023,1,1) + timedelta(days=np.random.randint(0,365)) for _ in range(n_users)],
    })

    data["last_active_date"] = data["signup_date"] + pd.to_timedelta(np.random.randint(1, 365, n_users), unit='d')

    data["transaction_count"] = np.random.poisson(20, n_users)
    data["transaction_value"] = np.round(np.random.exponential(5000, n_users), 2)
    data["session_frequency"] = np.random.randint(1, 50, n_users)

    data["device_type"] = np.random.choice(["Android","iOS","Web"], n_users, p=[0.6,0.3,0.1])
    data["location"] = np.random.choice(["Tier1","Tier2","Tier3"], n_users)

    data["payment_type"] = np.random.choice(["UPI","Wallet","Card"], n_users)

    data["failed_transactions"] = np.random.poisson(2, n_users)
    data["support_tickets"] = np.random.poisson(1, n_users)

    data["marketing_exposure"] = np.random.choice([0,1], n_users)

    data["tenure_days"] = (data["last_active_date"] - data["signup_date"]).dt.days

    today = datetime(2024,1,1)
    data["days_since_last_active"] = (today - data["last_active_date"]).dt.days

    data["churn"] = (data["days_since_last_active"] > 30).astype(int)

    os.makedirs("data/raw", exist_ok=True)
    data.to_csv("data/raw/fintech_churn.csv", index=False)

if __name__ == "__main__":
    generate_data()
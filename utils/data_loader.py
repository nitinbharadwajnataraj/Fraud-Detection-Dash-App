import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="static/dataset/fraud_oracle.csv"):
    df = pd.read_csv(path)

    # Drop unique identifiers
    df = df.drop(columns=["PolicyNumber", "RepNumber"], errors="ignore")

    # Separate features and target
    X = df.drop("FraudFound_P", axis=1)
    y = df["FraudFound_P"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

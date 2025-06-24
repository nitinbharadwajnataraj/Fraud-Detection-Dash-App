import yaml
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from models import get_model
from utils.data_loader import load_data
from utils.evaluator import evaluate_model

def build_encoder(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipeline, categorical_cols),
        ("num", num_pipeline, numerical_cols)
    ])

    return preprocessor

def train_all_models(config_dir="configs", result_path="results/model_metrics.json", encoder_path="results/encoder.pkl"):
    # Load raw data
    X_train, X_test, y_train, y_test = load_data()

    # Save raw input column names for future prediction requests
    raw_columns = X_train.columns.tolist()
    with open("results/encoder_columns.json", "w") as f:
        json.dump(raw_columns, f)
    print("✅ Saved raw input columns to encoder_columns.json")

    print("🔍 Class distribution BEFORE SMOTE:")
    print(pd.Series(y_train).value_counts())

    # Build and fit encoder
    encoder = build_encoder(pd.concat([X_train, X_test]))
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # Save encoder
    joblib.dump(encoder, encoder_path)
    print(f"✅ Encoder saved to {encoder_path}")

    metrics_dict = {}
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for cfg_file in Path(config_dir).glob("*.yaml"):
        with open(cfg_file, "r") as f:
            config = yaml.safe_load(f)

        model_name = config["model_name"]
        params = config["params"]
        model = get_model(model_name, params)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_encoded, y_train)

        print(f"\n🧪 Training '{model_name}'...")
        print("✅ Class distribution AFTER SMOTE:")
        print(pd.Series(y_train_res).value_counts())

        # Cross-validation F1 score
        y_cv_pred = cross_val_predict(model, X_train_encoded, y_train, cv=5)
        f1 = f1_score(y_train, y_cv_pred)

        # Train and predict
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_encoded)

        # Evaluation
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = evaluate_model(y_test, y_pred)
        metrics.update({
            "cv_f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "fraud_precision": round(report['1']['precision'], 4),
            "fraud_recall": round(report['1']['recall'], 4),
            "fraud_f1": round(report['1']['f1-score'], 4)
        })

        metrics_dict[model_name] = metrics
        model_path = results_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Saved model '{model_name}' to {model_path}")

    # Save metrics
    with open(result_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Model metrics saved to {result_path}")

if __name__ == "__main__":
    train_all_models()

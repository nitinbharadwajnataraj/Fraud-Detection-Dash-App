import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


DATA_PATH = "static/dataset/fraud_oracle.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

def get_data_overview(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "nulls": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "target_distribution": df["FraudFound_P"].value_counts().to_dict()
    }

def get_categorical_distribution(df, column):
    return px.histogram(df, x=column, color="FraudFound_P", barmode="group")

def get_numeric_distribution(df, column):
    return px.box(df, y=column, color="FraudFound_P")

def get_correlation_heatmap(df):
    corr_df = df.select_dtypes(include="number").corr()
    fig = px.imshow(corr_df, text_auto=True, title="Correlation Matrix")
    return fig

def get_full_correlation_heatmap(df):
    df_copy = df.copy()
    
    # Encode categoricals
    for col in df_copy.select_dtypes(include="object").columns:
        df_copy[col] = LabelEncoder().fit_transform(df_copy[col])
    
    corr_df = df_copy.corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        title="Correlation Matrix",
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        width=750,  # Increase width
        height=1000,  # Increase height
        margin=dict(l=50, r=50, t=50, b=50),  # More padding for labels
        xaxis_tickangle=45,
        font=dict(size=12)
    )
    return fig

def get_top_feature_importance_plot(df, target="FraudFound_P", top_n=10):
    df_copy = df.copy()
    
    # Drop rows with missing values (or you can impute)
    df_copy = df_copy.dropna()
    
    # Encode categorical features
    for col in df_copy.select_dtypes(include="object").columns:
        df_copy[col] = LabelEncoder().fit_transform(df_copy[col])

    # Split features and target
    X = df_copy.drop(columns=["PolicyNumber", "RepNumber","FraudFound_P"])
    y = df_copy[target]

    # Train a Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_
    feat_names = X.columns
    top_indices = importances.argsort()[-top_n:][::-1]

    # Plot
    fig = px.bar(
        x=[feat_names[i] for i in top_indices],
        y=[importances[i] for i in top_indices],
        labels={'x': 'Feature', 'y': 'Importance'},
        title=f"Top {top_n} Important Features Predicting '{target}'"
    )
    return fig

def get_top_feature_importance_PI(df, target="FraudFound_P", top_n=10):
    df_copy = df.copy()

    # Drop missing values (or use imputation)
    df_copy = df_copy.dropna()

    # Encode categorical columns
    for col in df_copy.select_dtypes(include="object").columns:
        df_copy[col] = LabelEncoder().fit_transform(df_copy[col])

    # Split features and target
    X = df_copy.drop(columns=["PolicyNumber", "RepNumber","FraudFound_P"])
    y = df_copy[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Compute permutation importances on test set
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": result.importances_mean
    })

    top_features = importance_df.sort_values(by="importance", ascending=False).head(top_n)

    # Plot
    fig = px.bar(
        top_features.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Permutation Importance", "feature": "Feature"},
        title=f"Top {top_n} Features Influencing Fraud Prediction (Permutation Importance)"
    )
    fig.update_layout(yaxis=dict(tickfont=dict(size=12)))
    return fig
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def get_model(model_name, params):
    if model_name == "LogisticRegression":
        return LogisticRegression(**params)
    elif model_name == "RandomForest":
        return RandomForestClassifier(**params)
    elif model_name == "XGBoost":
        return XGBClassifier(**params)
    elif model_name == "DecisionTree":
        return DecisionTreeClassifier(**params)
    elif model_name == "NaiveBayes":
        return GaussianNB(**params)
    elif model_name == "MLPClassifier":
        return MLPClassifier(**params)
    else:
        raise ValueError(f"Model '{model_name}' not implemented.")

"""
Classical ML models for stress level classification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    ),
    "SVM (RBF)": SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=42,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=3,
        metric="euclidean",
        weights="distance",
    ),
}

def train_all_ml_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    trained = {}
    for name, model in MODEL_REGISTRY.items():
        clf = type(model)(**model.get_params())
        clf.fit(X_train, y_train)
        trained[name] = clf
    return trained

def evaluate_models_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    records = []
    for name, model in MODEL_REGISTRY.items():
        clf = type(model)(**model.get_params())
        results = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        record = {"Model": name}
        for metric in scoring:
            key = f"test_{metric}"
            label = metric.replace("_macro", "").replace("_", " ").title()
            record[f"{label} Mean"] = round(results[key].mean(), 4)
            record[f"{label} Std"] = round(results[key].std(), 4)
        records.append(record)

    return pd.DataFrame(records)

"""
Classical ML models for stress level classification.

Models: Random Forest, XGBoost, SVM (RBF), KNN
Evaluation: Stratified K-Fold cross-validation
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


def train_all_ml_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Train all models in MODEL_REGISTRY, return dict[name → fitted model]."""
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
    """Run stratified K-fold CV on all ML models, return results DataFrame."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    records = []
    for name, model in MODEL_REGISTRY.items():
        clf = type(model)(**model.get_params())
        results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
        record = {"Model": name}
        for metric in scoring:
            label = metric.replace("_macro", "").replace("_", " ").title()
            record[f"Train {label}"] = round(results[f"train_{metric}"].mean(), 4)
            record[f"Test {label}"] = round(results[f"test_{metric}"].mean(), 4)
            
        train_acc = record["Train Accuracy"]
        test_acc = record["Test Accuracy"]
        
        if train_acc - test_acc > 0.05:
            record["Fit Status"] = "Overfit"
        elif train_acc < 0.85 and test_acc < 0.85:
            record["Fit Status"] = "Underfit"
        else:
            record["Fit Status"] = "Optimal"
            
        records.append(record)

    return pd.DataFrame(records)

"""
SHAP-based explainability for all model types.

- TreeExplainer  → Random Forest, XGBoost
- DeepExplainer  → MLP (PyTorch)
- KernelExplainer → SVM, KNN (model-agnostic)
"""

import numpy as np
import shap
import torch


def _background(X_train: np.ndarray, n: int = 50) -> np.ndarray:
    """Sample background data for SHAP explainers."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
    return X_train[idx]


def compute_shap_importance(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_bg: int = 50,
) -> np.ndarray:
    """
    Compute mean |SHAP| feature importance for any supported model.
    Returns 1-D array of shape (n_features,).
    """
    bg = _background(X_train, n=n_bg)
    sample = X_test[: min(50, len(X_test))]

    if model_name in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=(0, 1))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

    elif model_name == "MLP (PyTorch)":
        model.eval()
        bg_tensor = torch.tensor(bg, dtype=torch.float32)
        explainer = shap.DeepExplainer(model, bg_tensor)
        shap_values = explainer.shap_values(
            torch.tensor(sample, dtype=torch.float32)
        )
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=(0, 1))

    else:
        # Model-agnostic fallback (SVM, KNN)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(sample, nsamples=80, silent=True)
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=(0, 1))

    return mean_abs

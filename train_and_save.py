"""
Run locally once to generate serialized models, then commit the models/ folder.
This eliminates cold-start training on Streamlit Cloud.
"""

import os
import joblib
import torch
from src.preprocess import load_and_prepare
from src.ml_models import train_all_ml_models
from src.dl_model import train_mlp

os.makedirs("models", exist_ok=True)

data = load_and_prepare("data/SaYoPillow.csv")

# Train & save ML models
ml_models = train_all_ml_models(data["X_train"], data["y_train"])
name_map = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "SVM (RBF)": "svm_rbf.joblib",
    "KNN": "knn.joblib",
}
for name, model in ml_models.items():
    path = os.path.join("models", name_map[name])
    joblib.dump(model, path)
    print(f"Saved {path}")

# Train & save MLP
mlp = train_mlp(data["X_train"], data["y_train"], data["X_test"], data["y_test"])
torch.save(mlp.state_dict(), "models/mlp.pt")
print("Saved models/mlp.pt")

print("\nDone. Commit the models/ folder and push.")

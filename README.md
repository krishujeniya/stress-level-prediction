# Stress Level Prediction

Five-class physiological stress classifier trained on biosignals captured during sleep.

**Dataset**: [SaYoPillow](https://www.kaggle.com/datasets/laavanya/stress-level-prediction) — Rachakonda et al., IEEE Transactions on Consumer Electronics, 2021

## Features

| Code | Feature | Unit |
|------|---------|------|
| `sr` | Snoring Rate | — |
| `rr` | Respiration Rate | breaths/min |
| `t` | Body Temperature | °F |
| `lm` | Limb Movement Rate | — |
| `bo` | Blood Oxygen | % |
| `rem` | Eye Movement (REM) | — |
| `sh` | Hours of Sleep | hours |
| `hr` | Heart Rate | BPM |

**Target** (`sl`): 0 = Low/Normal, 1 = Medium Low, 2 = Medium, 3 = Medium High, 4 = High

## Models

- **Random Forest** (300 trees)
- **XGBoost** (300 rounds, depth 6)
- **SVM (RBF)** (C=10, gamma=scale)
- **KNN** (k=3, distance-weighted)
- **MLP (PyTorch)** (64→32→5, BatchNorm, Dropout, CosineAnnealing)

## Project Structure

```
├── app.py                 # Streamlit web application
├── train_and_save.py      # Local training script
├── train_colab.ipynb      # Google Colab training notebook
├── runtime.txt            # Python 3.12 for Streamlit Cloud
├── requirements.txt       # Pinned dependencies
├── data/
│   └── SaYoPillow.csv     # Dataset (630 samples)
├── src/
│   ├── preprocess.py      # Data loading, cleaning, scaling
│   ├── ml_models.py       # ML model registry & CV evaluation
│   ├── dl_model.py        # PyTorch MLP
│   └── explainability.py  # SHAP-based feature importance
└── models/                # Serialized trained models (generated)
    ├── random_forest.joblib
    ├── xgboost.joblib
    ├── svm_rbf.joblib
    ├── knn.joblib
    └── mlp.pt
```

## Quick Start

### Option A: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krishujeniya/stress-level-prediction/blob/main/train_colab.ipynb)

1. Open the Colab notebook
2. Enter your GitHub PAT (needs `repo` scope)
3. Run all cells → models train, serialize, and push to GitHub automatically

### Option B: Local

```bash
pip install -r requirements.txt
python train_and_save.py
streamlit run app.py
```

## Streamlit App

The app loads pre-trained models from `models/` for instant startup on Streamlit Cloud.

**Tabs**:
- **Predict** — Input biosignals via sliders, get stress prediction with confidence
- **Model Comparison** — 5-fold CV results, accuracy/F1 bar chart, radar plot
- **Explainability** — SHAP feature importance for any model
- **Dataset** — EDA: distributions, correlations, scatter plots

## License

MIT

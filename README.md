# Stress Level Prediction

A high-performance, five-class physiological stress classifier trained on biosignals captured during sleep. This project features a premium, glassmorphic Streamlit dashboard tailored for AI engineers, complete with advanced model diagnostics and SHAP explainability.

**Dataset**: [SaYoPillow](https://www.kaggle.com/datasets/laavanya/stress-level-prediction) — Rachakonda et al., IEEE Transactions on Consumer Electronics, 2021

## Features & Biosignals

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

## Evaluated Models

- **Random Forest** (300 trees)
- **XGBoost** (300 rounds, depth 6)
- **SVM (RBF)** (C=10, gamma=scale)
- **KNN** (k=3, distance-weighted)
- **MLP (PyTorch)** (64→32→5, BatchNorm, Dropout, CosineAnnealing)

## Project Structure

```
├── app.py                 # Streamlit web application (Glassmorphic UI)
├── train_and_save.py      # Local training script
├── train_colab.ipynb      # Sanitized Google Colab training notebook
├── runtime.txt            # Python 3.12 for Streamlit Cloud
├── requirements.txt       # Pinned dependencies
├── data/
│   └── SaYoPillow.csv     # Dataset (630 samples)
├── src/
│   ├── preprocess.py      # Data loading, cleaning, scaling, tooltips
│   ├── ml_models.py       # ML model registry & cross-validation metrics
│   ├── dl_model.py        # PyTorch MLP
│   └── explainability.py  # SHAP-based feature importance
└── models/                # Serialized trained models (generated)
```

## Quick Start (Local with `uv`)

This project uses the extremely fast `uv` package manager for Python.

```bash
uv venv
uv pip install -r requirements.txt
uv run python train_and_save.py
uv run streamlit run app.py
```

## Dashboard Features

The Streamlit app has been ruthlessly optimized for an AI engineering workflow, stripping away univariate clutter and focusing strictly on multivariate analysis and model performance.

- **Predict** — Input biosignals via intuitive sliders with embedded scientific tooltips (`?`).
- **Model Comparison** — View comprehensive Train/Test evaluation metrics (Accuracy, Precision, Recall, F1) alongside automatic **Fit Status** classification (Underfit, Optimal, Overfit). It also includes dynamically rendered SHAP feature importance charts.
- **Dataset** — Minimalist EDA focusing strictly on high-level data topology and a Pearson Correlation Matrix.

## Google Colab

The repository includes a pristine, privacy-first `train_colab.ipynb` notebook. It is strictly scoped to loading the dataset, executing the PyTorch/XGBoost training loops, and serializing the artifacts without any external deployment bloat.

## License

MIT

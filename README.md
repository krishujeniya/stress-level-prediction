# Stress Level Prediction

Five-class physiological stress classifier.  
Dataset: SaYoPillow — Rachakonda et al., IEEE TCE 2021.  
Kaggle reference: https://www.kaggle.com/code/krishujeniya/stress-level-prediction

---

## Setup

### 1. Get the dataset

Download `SaYoPillow.csv` from Kaggle:  
https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep

Place it at:
```
data/SaYoPillow.csv
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this entire repository to GitHub (public or private).
2. Go to https://share.streamlit.io → New app.
3. Set:
   - Repository: your repo
   - Branch: main
   - Main file path: `app.py`
4. Click Deploy.

The `requirements.txt` handles CPU-only PyTorch automatically.

---

## File Structure

```
stress-level-prediction/
├── app.py                    Streamlit entry point
├── requirements.txt
├── data/
│   └── SaYoPillow.csv        Dataset (download from Kaggle)
├── src/
│   ├── __init__.py
│   ├── preprocess.py         Data loading, cleaning, scaling, splitting
│   ├── ml_models.py          RF, XGBoost, SVM, KNN — train + CV evaluation
│   ├── dl_model.py           PyTorch MLP — StressMLP architecture
│   └── explainability.py     SHAP wrappers for all model types
└── README.md
```

---

## Models

| Model         | Type | SHAP Method      |
|---------------|------|------------------|
| Random Forest | ML   | TreeExplainer    |
| XGBoost       | ML   | TreeExplainer    |
| SVM (RBF)     | ML   | KernelExplainer  |
| KNN           | ML   | KernelExplainer  |
| MLP (PyTorch) | DL   | DeepExplainer    |

---

## Dataset Features

| Column | Description                  |
|--------|------------------------------|
| sr     | Snoring Rate                 |
| rr     | Respiration Rate             |
| t      | Body Temperature (F)         |
| lm     | Limb Movement Rate           |
| bo     | Blood Oxygen (%)             |
| rem    | Eye Movement (REM)           |
| sh     | Hours of Sleep               |
| hr     | Heart Rate (BPM)             |
| sl     | Stress Level — target (0-4)  |

Stress labels: 0=Low/Normal, 1=Medium Low, 2=Medium, 3=Medium High, 4=High

---

## Citation

```
L. Rachakonda, A. K. Bapatla, S. P. Mohanty, and E. Kougianos,
"SaYoPillow: Blockchain-Integrated Privacy-Assured IoMT Framework
for Stress Management Considering Sleeping Habits",
IEEE Transactions on Consumer Electronics (TCE), Vol. 67, No. 1, Feb 2021, pp. 20-29.
```

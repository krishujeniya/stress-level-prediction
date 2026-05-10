"""
Preprocessing pipeline for SaYoPillow stress detection dataset.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_COLS = ["sr", "rr", "t", "lm", "bo", "rem", "sh", "hr"]
TARGET_COL = "sl"

FEATURE_DISPLAY = {
    "sr": "Snoring Rate",
    "rr": "Respiration Rate",
    "t": "Body Temperature (F)",
    "lm": "Limb Movement Rate",
    "bo": "Blood Oxygen (%)",
    "rem": "Eye Movement (REM)",
    "sh": "Hours of Sleep",
    "hr": "Heart Rate (BPM)",
}

LABEL_MAP = {
    0: "Low / Normal",
    1: "Medium Low",
    2: "Medium",
    3: "Medium High",
    4: "High",
}

STRESS_COLORS = {
    0: "#2d6a4f",
    1: "#74c69d",
    2: "#f4a261",
    3: "#e76f51",
    4: "#c1121f",
}

FEATURE_RANGES = {
    "sr": (40.0, 100.0, 65.0, 0.5),
    "rr": (14.0, 30.0, 20.0, 0.5),
    "t": (84.0, 100.0, 98.0, 0.1),
    "lm": (4.0, 20.0, 8.0, 0.5),
    "bo": (78.0, 100.0, 96.0, 0.1),
    "rem": (50.0, 120.0, 80.0, 1.0),
    "sh": (1.0, 9.0, 7.0, 0.5),
    "hr": (50.0, 95.0, 70.0, 1.0),
}

DATASET_URL = "https://raw.githubusercontent.com/krishujeniya/stress-level-prediction/main/data/SaYoPillow.csv"

def _ensure_dataset(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        urllib.request.urlretrieve(DATASET_URL, path)
    except Exception as exc:
        raise RuntimeError(
            f"Dataset not found at {path} and auto-download failed. "
            f"Please download SaYoPillow.csv manually from Kaggle and place it in {os.path.dirname(path)}/"
        ) from exc

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    for old, new in {
        "sr.1": "sh",
        "sleeping hours": "sh",
        "snoring rate": "sr",
        "respiration rate": "rr",
        "body temperature": "t",
        "limb movement": "lm",
        "blood oxygen": "bo",
        "eye movement": "rem",
        "heart rate": "hr",
        "stress level": "sl",
    }.items():
        if old in df.columns:
            rename_map[old] = new
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def load_and_prepare(path: str, test_size: float = 0.2, random_state: int = 42) -> dict:
    _ensure_dataset(path)
    df = pd.read_csv(path)
    df = _normalise_columns(df)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns after normalisation: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[FEATURE_COLS + [TARGET_COL]].copy()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        "df": df,
        "X_scaled": X_scaled,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }

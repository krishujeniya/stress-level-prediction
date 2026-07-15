"""
Preprocessing pipeline for SaYoPillow stress detection dataset.
Dataset: 630 samples, 8 physiological features, 5-class stress target (0–4).
Reference: Rachakonda et al., IEEE Transactions on Consumer Electronics, 2021.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Column names in the raw CSV ──────────────────────────────────────────────
# sr  = snoring rate          rr  = respiration rate
# t   = body temperature (F)  lm  = limb movement rate
# bo  = blood oxygen (%)      rem = rapid eye movement
# sr.1 (renamed → sh) = sleeping hours   hr = heart rate (BPM)
# sl  = stress level (target, 0–4)

FEATURE_COLS = ["sr", "rr", "t", "lm", "bo", "rem", "sh", "hr"]
TARGET_COL = "sl"

FEATURE_DISPLAY = {
    "sr": "Snoring Rate",
    "rr": "Respiration Rate",
    "t": "Body Temperature (°F)",
    "lm": "Limb Movement Rate",
    "bo": "Blood Oxygen (%)",
    "rem": "Eye Movement (REM)",
    "sh": "Hours of Sleep",
    "hr": "Heart Rate (BPM)",
}

FEATURE_DESC = {
    "sr": "Acoustic measurement of snoring frequency. Elevated rates often correlate with sleep apnea or elevated stress.",
    "rr": "Number of breaths per minute. Higher rates during sleep indicate sympathetic nervous system arousal.",
    "t": "Core body temperature. Should drop slightly during deep sleep; elevated temps disrupt sleep architecture.",
    "lm": "Frequency of limb twitches. Restless legs or high movement indicates fragmented, non-restorative sleep.",
    "bo": "SpO2 levels. Drops in oxygen saturation trigger cortisol release, indicating high physiological stress.",
    "rem": "Rapid Eye Movement phase duration. Essential for cognitive recovery; stress often suppresses REM sleep.",
    "sh": "Total continuous hours of sleep recorded. Acute deprivation directly elevates daytime stress markers.",
    "hr": "Resting heart rate in BPM. Elevated nocturnal heart rate is a primary indicator of physiological strain.",
}

LABEL_MAP = {
    0: "Low / Normal",
    1: "Medium Low",
    2: "Medium",
    3: "Medium High",
    4: "High",
}

STRESS_COLORS = {
    0: "#ffffff",
    1: "#d4d4d8",
    2: "#a1a1aa",
    3: "#71717a",
    4: "#3f3f46",
}

# (min, max, default, step) for Streamlit sliders
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

DATASET_URL = (
    "https://raw.githubusercontent.com/krishujeniya/"
    "stress-level-prediction/main/data/SaYoPillow.csv"
)


def _ensure_dataset(path: str) -> None:
    """Download SaYoPillow.csv from GitHub if not present locally."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        urllib.request.urlretrieve(DATASET_URL, path)
    except Exception as exc:
        raise RuntimeError(
            f"Dataset not found at {path} and auto-download failed.\n"
            f"Please place SaYoPillow.csv in {os.path.dirname(path)}/"
        ) from exc


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names to short-form keys."""
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


def load_and_prepare(
    path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Load CSV → normalise → clean → scale → split.

    Returns dict with keys:
        df, X_scaled, y, X_train, X_test, y_train, y_test, scaler
    """
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

"""
Train a Random Forest surrogate model to approximate the LLM's prediction
function using features extracted in prev step.

- Classification report printed to console
- Trained RF model saved as a .pkl file
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config


NON_FEATURE_COLS = {"audio_path", "pred_label", "fake_score", "real_score"}


def load_data(cfg: dict):
    csv_path = cfg["features"]["output_csv"]
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {csv_path}")

    import numpy as np
    df = df.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

    drop = [c for c in NON_FEATURE_COLS if c in df.columns and c != "pred_label"]
    X = df.drop(columns=drop + ["pred_label"])
    y = df["pred_label"].astype(int)
    return X, y


def main() -> None:
    cfg = load_config()
    sc  = cfg["surrogate"]

    X, y = load_data(cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=sc["test_size"],
        random_state=sc["random_state"],
    )

    print(
        f"\nTraining Random Forest"
        f" (n_estimators={sc['n_estimators']}, max_depth={sc['max_depth']}) …"
    )

    rf = RandomForestClassifier(
        n_estimators=sc["n_estimators"],
        max_depth=sc["max_depth"],
        random_state=sc["random_state"],
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\nClassification Report (surrogate on held-out test split):")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    model_path = Path(sc["model_path"])
    joblib.dump(rf, model_path)
    print(f"Surrogate model saved - {model_path}")

    col_path = model_path.parent / (model_path.stem + "_feature_cols.pkl")
    joblib.dump(X.columns.tolist(), col_path)
    print(f"Feature column list saved - {col_path}")


if __name__ == "__main__":
    main()

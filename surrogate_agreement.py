from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
)

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config


NON_FEATURE_COLS = {"audio_path", "pred_label", "fake_score", "real_score"}


def main() -> None:
    cfg = load_config()
    model_name = cfg["model"]["name"]

    print("Loading feature CSV and surrogate model …")
    df = pd.read_csv(cfg["features"]["output_csv"])
    df = df.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

    y_llm = df["pred_label"].astype(int)
    drop   = [c for c in NON_FEATURE_COLS if c in df.columns and c != "pred_label"]
    X      = df.drop(columns=drop + ["pred_label"])

    rf = joblib.load(cfg["surrogate"]["model_path"])

    y_rf     = rf.predict(X)
    rf_probs = rf.predict_proba(X)[:, 1] 

    acc   = accuracy_score(y_llm, y_rf)
    mcc   = matthews_corrcoef(y_llm, y_rf)
    kappa = cohen_kappa_score(y_llm, y_rf)

    print(f"\n── {model_name} – Surrogate–LLM Agreement ──")
    print(f"Accuracy:      {acc:.4f}")
    print(f"MCC:           {mcc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    print("\nClassification Report (RF predictions vs LLM labels):")
    print(classification_report(y_llm, y_rf, target_names=["Real", "Fake"]))

    cm = confusion_matrix(y_llm, y_rf)
    print("Confusion Matrix (LLM rows, RF columns):")
    print(f"              RF Real  RF Fake")
    print(f"LLM Real    {cm[0, 0]:8d}  {cm[0, 1]:8d}")
    print(f"LLM Fake    {cm[1, 0]:8d}  {cm[1, 1]:8d}")

    pearson_r,  _ = pearsonr(rf_probs, y_llm)
    spearman_r, _ = spearmanr(rf_probs, y_llm)

    print("\nProbability-Level Correlation (RF P(fake) vs LLM label):")
    print(f"Pearson r:   {pearson_r:.4f}")
    print(f"Spearman ρ:  {spearman_r:.4f}")


if __name__ == "__main__":
    main()

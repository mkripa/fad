# Reqd: features.output_csv, surrogate.model_path

from __future__ import annotations
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lime.lime_tabular
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config

NON_FEATURE_COLS = {"audio_path", "pred_label", "fake_score", "real_score"}

def load_splits(cfg: dict):
    sc = cfg["surrogate"]
    df = pd.read_csv(cfg["features"]["output_csv"])
    df = df.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
    audio_paths = df.get("audio_path", pd.Series([""] * len(df)))
    drop = [c for c in NON_FEATURE_COLS if c in df.columns and c != "pred_label"]
    df_feat = df.drop(columns=drop + ["pred_label"])
    X = df_feat
    y = df["pred_label"].astype(int)

    X_train, X_test, y_train, y_test, audio_train, audio_test = train_test_split(
        X, y, audio_paths,
        stratify=y,
        test_size=sc["test_size"],
        random_state=sc["random_state"],
    )

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    audio_test = audio_test.reset_index(drop=True)

    return X_train, X_test, y_test, audio_test


def main() -> None:
    cfg = load_config()
    ex_cfg = cfg["explainability"]
    model_name = cfg["model"]["name"]
    lime_dir = Path(cfg["outputs"]["lime_dir"])
    prefix = cfg["outputs"]["file_prefix"]
    indices = ex_cfg.get("lime_sample_indices", [])
    if not indices:
        print("No lime_sample_indices specified in config — skipping LIME step.")
        return
    print("Loading surrogate model …")
    rf = joblib.load(cfg["surrogate"]["model_path"])

    print("Reconstructing train/test splits …")
    X_train, X_test, y_test, audio_test = load_splits(cfg)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["Real", "Fake"],
        discretize_continuous=True,
    )

    for idx in indices:
        if idx >= len(X_test):
            print(f"[WARN] Index {idx} is out of range (test set has {len(X_test)} samples) — skipping.")
            continue

        audio_file = audio_test.iloc[idx] if len(audio_test) > idx else "n/a"
        label_val  = y_test.iloc[idx]
        label_name = "Fake" if label_val == 1 else "Real"

        print(f"\nSample #{idx}")
        print(f"  Audio : {audio_file}")
        print(f"  Label : {label_name}")

        lime_exp = lime_explainer.explain_instance(
            X_test.iloc[idx].values,
            rf.predict_proba,
            num_features=ex_cfg["lime_num_features"],
        )

        html_path = lime_dir / f"{prefix}_{model_name}_lime_idx{idx}.html"
        with open(html_path, "w") as fh:
            fh.write(lime_exp.as_html())
        print(f"  Saved - {html_path}")

        print("  Top features:")
        for feat, weight in lime_exp.as_list():
            print(f"    {feat:50s}  {weight:+.4f}")

    print("\nLIME explanations complete.")


if __name__ == "__main__":
    main()

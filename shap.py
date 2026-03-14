# Reqd: features.output_csv, surrogate.model_path

from __future__ import annotations
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config, output_path

NON_FEATURE_COLS = {"audio_path", "pred_label", "fake_score", "real_score"}

def load_test_split(cfg: dict):
    sc = cfg["surrogate"]
    df = pd.read_csv(cfg["features"]["output_csv"])
    df = df.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

    drop = [c for c in NON_FEATURE_COLS if c in df.columns and c != "pred_label"]
    X = df.drop(columns=drop + ["pred_label"])
    y = df["pred_label"].astype(int)

    _, X_test, _, _ = train_test_split(
        X, y,
        stratify=y,
        test_size=sc["test_size"],
        random_state=sc["random_state"],
    )
    return X_test.reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    ex_cfg = cfg["explainability"]
    model_name = cfg["model"]["name"]

    print("Loading surrogate model …")
    rf = joblib.load(cfg["surrogate"]["model_path"])

    print("Reconstructing test split …")
    X_test = load_test_split(cfg)

    print("Computing SHAP values (TreeExplainer) …")
    explainer  = shap.TreeExplainer(rf)
    shap_vals  = explainer.shap_values(X_test)         
    shap_array = np.array(shap_vals)

    shap_fake = shap_array[:, :, 1]

    mean_abs = np.abs(shap_fake).mean(axis=0)
    shap_df = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
    )

    top_n = ex_cfg["shap_top_n"]
    print(f"\nTop {top_n} SHAP features (mean |SHAP|, class=Fake):")
    print(shap_df.head(top_n).to_string(index=False))

    max_display = ex_cfg["shap_max_display"]
    save_path   = output_path(cfg, "shap_summary.png")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_fake,
        X_test,
        max_display=max_display,
        show=False,
    )
    plt.title(f"{model_name} – SHAP Summary (class: Fake, top {max_display})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSHAP summary plot saved - {save_path}")

    csv_path = save_path.with_suffix(".csv")
    shap_df.to_csv(csv_path, index=False)
    print(f"SHAP feature table saved -> {csv_path}")


if __name__ == "__main__":
    main()

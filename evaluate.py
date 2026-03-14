#pred_source.mode == "json_scores" # {audio_path: [fake_score, real_score]}
# (fake_score > real_score  =>  "fake")
#pred_source.mode == "csv_labels" # CSV with [audio_path, pred_label] columns.
# (pred_label - 0 (real) or 1 (fake))
# ground_truth.mode == "jsonl" / "folder"

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config, output_path

def load_predictions(cfg: dict) -> dict[str, int]:
    mode = cfg["pred_source"]["mode"]
    path = cfg["pred_source"]["path"]

    if mode == "json_scores":
        with open(path) as fh:
            raw = json.load(fh)
        return {
            audio: (1 if scores[0] > scores[1] else 0)
            for audio, scores in raw.items()
        }

    if mode == "csv_labels":
        import pandas as pd

        df = pd.read_csv(path)
        return dict(zip(df["audio_path"], df["pred_label"].astype(int)))

    raise ValueError(f"Unknown pred_source.mode: {mode!r}")


def load_ground_truth(cfg: dict) -> dict[str, int]:
    mode = cfg["ground_truth"]["mode"]
    path = cfg["ground_truth"]["path"]
    max_per_class: int | None = cfg["sampling"]["max_per_class"]

    counts = {"real": 0, "fake": 0}
    gt: dict[str, int] = {}

    if mode == "jsonl":
        with open(path) as fh:
            for line in fh:
                if max_per_class and all(v >= max_per_class for v in counts.values()):
                    break
                data = json.loads(line)
                audio = data["messages"][0]["audio"]
                label = data["messages"][1]["content"].strip(".").lower()
                if max_per_class and counts[label] >= max_per_class:
                    continue
                gt[audio] = 1 if label == "fake" else 0
                counts[label] += 1
        return gt

    if mode == "folder":
        for label in ("real", "fake"):
            folder = Path(path) / label
            for fname in sorted(os.listdir(folder)):
                if max_per_class and counts[label] >= max_per_class:
                    break
                if fname.lower().endswith(".wav"):
                    gt[str(folder / fname)] = 1 if label == "fake" else 0
                    counts[label] += 1
        return gt
    raise ValueError(f"Unknown ground_truth.mode: {mode!r}")

def evaluate(
    preds: dict[str, int],
    gt: dict[str, int],
    model_name: str,
) -> tuple[list[int], list[int]]:
    y_true, y_pred = [], []
    for audio, true_label in gt.items():
        if audio in preds:
            y_true.append(true_label)
            y_pred.append(preds[audio])

    if not y_true:
        raise RuntimeError(
            "No matching audio paths between predictions and ground truth. "
            "Check that file paths are consistent across both files."
        )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)

    print(f"\n── {model_name} Detection Results ({'–'.join(str(s) for s in [len(y_true)])} samples) ──")
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
    print(f"{'':16s} Pred Real  Pred Fake")
    print(f"{'Actual Real':16s} {tn:8d}   {fp:8d}")
    print(f"{'Actual Fake':16s} {fn:8d}   {tp:8d}")
    print(f"\nAccuracy  = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall    = {rec:.4f}")
    print(f"F1 Score  = {f1:.4f}")
    print(f"MCC       = {mcc:.4f}")

    return y_true, y_pred, cm


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, model_name: str) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"{model_name} – Audio Deepfake Detection")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved - {save_path}")

def main() -> None:
    cfg = load_config()
    model_name = cfg["model"]["name"]

    print("Loading predictions …")
    preds = load_predictions(cfg)
    print(f"  {len(preds):,} prediction entries loaded.")

    print("Loading ground truth …")
    gt = load_ground_truth(cfg)
    print(f"  {len(gt):,} ground-truth entries loaded.")

    y_true, y_pred, cm = evaluate(preds, gt, model_name)

    save_path = output_path(cfg, "confusion_matrix.png")
    plot_confusion_matrix(cm, save_path, model_name)


if __name__ == "__main__":
    main()

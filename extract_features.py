# Extract OpenSMILE features for audio files and write to a CSV along with pred.

# pred_source.mode == "json_scores"   {audio_path: [fake_score, real_score]}
# "csv_labels"    CSV with [audio_path, pred_label]

# Cols in addition to features
#  audio_path
#  pred_label - 0 (real) or 1 (fake)
#  fake_score (for json_scores mode)
#  real_score (for json_scores mode)


from __future__ import annotations
import json
import sys
from pathlib import Path
import opensmile
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config

def iter_entries(cfg: dict):
    mode = cfg["pred_source"]["mode"]
    path = cfg["pred_source"]["path"]
    max_pc: int | None = cfg["sampling"]["max_per_class"]
    counts = {0: 0, 1: 0}
    stop   = {0: False, 1: False}

    def _emit(audio_path, label, extra):
        nonlocal counts, stop
        if max_pc and counts[label] >= max_pc:
            stop[label] = True
            return None
        counts[label] += 1
        return (audio_path, label, extra)

    if mode == "json_scores":
        with open(path) as fh:
            raw = json.load(fh)
        for audio, scores in raw.items():
            if max_pc and all(stop.values()):
                break
            fake_s, real_s = scores
            label = 1 if fake_s > real_s else 0
            result = _emit(audio, label, {"fake_score": fake_s, "real_score": real_s})
            if result:
                yield result

    elif mode == "csv_labels":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if max_pc and all(stop.values()):
                break
            label = int(row["pred_label"])
            result = _emit(row["audio_path"], label, {})
            if result:
                yield result

    else:
        raise ValueError(f"Unknown pred_source.mode: {mode!r}")


def main() -> None:
    cfg = load_config()
    output_csv = Path(cfg["features"]["output_csv"])

    feature_set_name  = cfg["features"]["opensmile_feature_set"]
    feature_level_name = cfg["features"]["opensmile_feature_level"]

    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, feature_set_name),
        feature_level=getattr(opensmile.FeatureLevel, feature_level_name),
    )

    entries = list(iter_entries(cfg))
    print(f"\nProcessing {len(entries):,} audio files with OpenSMILE ({feature_set_name}) …")

    rows = []
    errors = 0

    for audio_path, pred_label, extra in tqdm(entries, desc="Extracting features"):
        try:
            feats = smile.process_file(audio_path)
            row = feats.iloc[0].to_dict()
            row["audio_path"] = audio_path
            row["pred_label"] = pred_label
            row.update(extra)
            rows.append(row)
        except Exception as exc:
            errors += 1
            print(f"\n[WARN] {audio_path}: {exc}")

    if not rows:
        raise RuntimeError("No features extracted — check audio paths.")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    real_count = (df["pred_label"] == 0).sum()
    fake_count = (df["pred_label"] == 1).sum()
    feat_cols  = len(df.columns) - len(["audio_path", "pred_label"] + list(extra.keys()))

    print(f"\nSaved {len(df):,} rows - {output_csv}")
    print(f"  Real(0)={real_count}, Fake(1)={fake_count}")
    print(f"  Feature columns: {feat_cols}")
    if errors:
        print(f"  Files skipped due to errors: {errors}")


if __name__ == "__main__":
    main()

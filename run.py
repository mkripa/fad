"""
Usage:
    python run_all.py --config config.yaml
    python run_all.py --config config.yaml --steps 1 2 3   # run only listed steps
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

STEPS = [
    ("evaluate",            "1 Compare model predictions against ground truth"),
    ("extract_features",    "2 Extract OpenSMILE features"),
    ("train_surrogate",     "3 Train Random Forest surrogate"),
    ("shap",                "4 SHAP explainability"),
    ("lime",                "5 LIME explanations"),
    ("surrogate_agreement", "6 Metrics"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Run the full ADD evaluation pipeline.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--force", action="store_true", help="Re-run all steps.")
    p.add_argument(
        "--steps",
        nargs="+",
        type=int,
        metavar="N",
        help="Only run these step numbers (1-based).",
    )
    return p.parse_args()


def run_step(module_name: str, description: str, args) -> None:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    t0 = time.time()
    sys.argv = [module_name + ".py", "--config", args.config]
    mod = importlib.import_module(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n  ✓ Done in {elapsed:.1f}s")


def main():
    args = parse_args()

    selected = set(args.steps) if args.steps else set(range(1, len(STEPS) + 1))

    print(f"\nADD Evaluation Pipeline  —  config: {args.config}")
    print(f"Steps to run: {sorted(selected)}")

    sys.path.insert(0, str(Path(__file__).parent))

    for i, (module, desc) in enumerate(STEPS, start=1):
        if i not in selected:
            print(f"\n[Step {i}] Skipped: {desc}")
            continue
        run_step(module, f"Step {i}: {desc}", args)

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

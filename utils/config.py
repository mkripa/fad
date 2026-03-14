from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import yaml

def load_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file.",
    )
    args, _ = parser.parse_known_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    for key in ("plots_dir", "lime_dir"):
        p = cfg.get("outputs", {}).get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)

    for dotted in ("features.output_csv", "surrogate.model_path"):
        section, field = dotted.split(".")
        p = cfg.get(section, {}).get(field)
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    return cfg

def output_path(cfg: dict, suffix: str) -> Path:
    plots_dir = Path(cfg["outputs"]["plots_dir"])
    prefix = cfg["outputs"]["file_prefix"]
    model = cfg["model"]["name"]
    return plots_dir / f"{prefix}_{model}_{suffix}"

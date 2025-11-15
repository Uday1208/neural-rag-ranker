# scripts/download_msmarco.py
"""
Utility script to download MS MARCO and generate a small preview file.

This script uses the Hugging Face Datasets library to stream the
`microsoft/ms_marco` v1.1 dataset, then writes a limited number of
examples to JSONL for quick inspection and debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root (which contains the 'src' package) is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from typing import Any, Dict

from datasets import load_dataset

from src.utils.config import load_config
from src.utils.timing import time_block  # <-- new import


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the MS MARCO download script.

    Returns:
        An argparse Namespace containing parsed command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Download MS MARCO and create a preview JSONL file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_msmarco.yaml",
        help=(
            "Path to the MS MARCO data configuration YAML file "
            "(merged with config/base.yaml)."
        ),
    )
    return parser.parse_args()


def download_msmarco_preview(config_path: str) -> None:
    """
    Download MS MARCO data and emit a small preview JSONL file to disk.

    Args:
        config_path:
            Path to the YAML configuration with data.msmarco settings.

    Notes:
        This function is intentionally conservative and only writes a
        limited number of rows for inspection; training code should use
        the Dataset wrappers directly instead of relying on this output.
    """
    cfg: Dict[str, Any] = load_config(config_path)

    msmarco_cfg: Dict[str, Any] = cfg["data"]["msmarco"]
    hf_name: str = msmarco_cfg["hf_dataset_name"]
    hf_config: str = msmarco_cfg["hf_config_name"]
    train_split: str = msmarco_cfg["train_split"]

    preview_num: int = int(msmarco_cfg.get("preview_num_examples", 1000))
    output_dir = Path(msmarco_cfg.get("output_dir", "./data/msmarco"))
    preview_filename: str = msmarco_cfg.get("preview_file", "preview_train.jsonl")

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / preview_filename

    with time_block("MS MARCO preview download+write"):
        print(
            f"[download_msmarco] Loading dataset {hf_name!r} (config={hf_config!r}, "
            f"split={train_split!r}) in streaming mode..."
        )
    
        # Streaming mode is memory-friendly for large datasets.
        dataset_iter = load_dataset(
            hf_name,
            hf_config,
            split=train_split,
            #streaming=True, # Non-streaming load: download the parquet once, then slice in-memory.
        )

        # Added for non-streaming mode
        # Limit to the desired preview size.
        num_to_take = min(preview_num, len(hf_ds))
        hf_ds = hf_ds.select(range(num_to_take))
    
        written = 0
        with preview_path.open("w", encoding="utf-8") as output_file:
            for example in dataset_iter:
                # We keep the preview simple and focused on relevant fields.
                row = {
                    "query_id": example.get("query_id"),
                    "query": example.get("query"),
                    "passages": example.get("passages"),
                }
                json_line = json.dumps(row, ensure_ascii=False)
                output_file.write(json_line + "\n")
    
                written += 1
                if written >= preview_num:
                    break
    
        print(
            f"[download_msmarco] Wrote {written} examples to {preview_path.as_posix()}"
        )


def main() -> None:
    """
    Entry point for running the MS MARCO download script from the CLI.

    This function parses arguments and triggers the preview download.
    """
    args = parse_args()
    download_msmarco_preview(args.config)


if __name__ == "__main__":
    main()

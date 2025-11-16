# scripts/run_bm25_baseline.py
"""
Run a BM25 baseline on the MS MARCO evaluation split.

This script builds a per-query BM25 retriever over the candidate
passages associated with each MS MARCO query, ranks them, and computes
IR metrics such as MRR@K, Recall@K, Precision@K, and nDCG@K.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root (containing 'src') is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.msmarco_dataset import MSMarcoRankingDataset
from src.evaluation.metrics import compute_metrics_at_ks
from src.retrieval.bm25 import BM25Retriever
from src.utils.config import load_config
from src.utils.device import seed_everything
from src.utils.timing import time_block


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the BM25 baseline script.

    Returns:
        Argparse Namespace containing parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Run a BM25 baseline on the MS MARCO evaluation split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_msmarco.yaml",
        help=(
            "Path to the YAML configuration file with data.msmarco and "
            "eval.msmarco_bm25 settings."
        ),
    )
    return parser.parse_args()


def run_bm25_baseline(config_path: str) -> None:
    """
    Load configuration, run BM25 ranking, and print IR metrics.

    Args:
        config_path:
            Path to the YAML configuration file.
    """
    cfg: Dict[str, Any] = load_config(config_path)

    msmarco_cfg: Dict[str, Any] = cfg["data"]["msmarco"]
    eval_cfg: Dict[str, Any] = cfg.get("eval", {}).get("msmarco_bm25", {})

    eval_split: str = msmarco_cfg.get("eval_split", "validation")
    max_eval_samples: int = int(msmarco_cfg.get("max_eval_samples", 10000))

    top_k: int = int(eval_cfg.get("top_k", 10))
    metrics_k: List[int] = [int(k) for k in eval_cfg.get("metrics_k", [1, 3, 5, 10])]

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    seed_everything(seed)

    print(
        f"[bm25] Using split={eval_split!r}, max_eval_samples={max_eval_samples}, "
        f"top_k={top_k}, metrics_k={metrics_k}, seed={seed}"
    )

    with time_block("load MS MARCO ranking eval dataset"):
        ranking_dataset = MSMarcoRankingDataset(
            split=eval_split,
            hf_dataset_name=msmarco_cfg["hf_dataset_name"],
            hf_config_name=msmarco_cfg["hf_config_name"],
            max_samples=max_eval_samples,
        )

    print(f"[bm25] Loaded {len(ranking_dataset)} queries with positives.")

    # qrels: qid -> doc_id -> relevance
    qrels: Dict[str, Dict[str, int]] = {}
    # run: qid -> ranked list of doc_ids
    run: Dict[str, List[str]] = {}

    with time_block("BM25 scoring and run construction"):
        for example in ranking_dataset:
            qid = example.query_id
            passages = example.passages
            labels = example.relevance_labels

            # Create per-query doc IDs (local to this query).
            doc_ids = [f"p{i}" for i in range(len(passages))]
            relevant_doc_ids = {
                doc_id for doc_id, rel in zip(doc_ids, labels) if rel > 0
            }

            if not relevant_doc_ids:
                # This should be rare due to filtering, but we guard anyway.
                continue

            qrels[qid] = {doc_id: 1 for doc_id in relevant_doc_ids}

            retriever = BM25Retriever(documents=passages, doc_ids=doc_ids)
            ranked = retriever.search(example.query, top_k=top_k)
            run[qid] = [doc_id for doc_id, _ in ranked]

    with time_block("metric computation"):
        metrics = compute_metrics_at_ks(qrels, run, ks=metrics_k)

    num_queries = len(qrels)
    print(f"[bm25] Evaluated {num_queries} queries with at least one relevant doc.\n")
    print("BM25 baseline metrics on MS MARCO eval split:")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")


def main() -> None:
    """
    Entry point when running this module as a script.

    This function parses arguments and executes the BM25 baseline run.
    """
    args = parse_args()
    run_bm25_baseline(args.config)


if __name__ == "__main__":
    main()

# scripts/train_biencoder.py
"""
Entry-point script for training the bi-encoder ranker on MS MARCO.

This script wires together configuration loading, dataset creation,
tokenization, model construction, optimizer/scheduler setup, and
the BiEncoderTrainer to run contrastive training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure project root is on sys.path for 'src' imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collators import BiEncoderCollator
from src.data.msmarco_dataset import MSMarcoTripletDataset
from src.models.biencoder import BiEncoderModel
from src.training.optim import create_optimizer_and_scheduler
from src.training.trainer_biencoder import BiEncoderTrainer
from src.utils.config import load_config
from src.utils.device import get_device, seed_everything
from src.utils.timing import time_block


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the bi-encoder training script.

    Returns:
        Argparse Namespace with the parsed configuration path.
    """
    parser = argparse.ArgumentParser(
        description="Train a bi-encoder neural ranker on MS MARCO."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_biencoder.yaml",
        help="Path to the bi-encoder training YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for launching bi-encoder training.

    This function orchestrates configuration loading, dataset and
    dataloader construction, model setup, and the training loop.
    """
    args = parse_args()
    cfg: Dict[str, Any] = load_config(args.config)

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    seed_everything(seed)

    device = get_device(
        prefer_gpu=bool(cfg.get("device", {}).get("use_cuda_if_available", True))
    )
    print(f"[train_biencoder] Using device: {device}")

    data_cfg = cfg["data"]["msmarco"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Datasets
    with time_block("build MS MARCO triplet train dataset"):
        train_dataset = MSMarcoTripletDataset(
            split=data_cfg["train_split"],
            hf_dataset_name=data_cfg["hf_dataset_name"],
            hf_config_name=data_cfg["hf_config_name"],
            max_samples=int(data_cfg.get("max_train_samples", 50000)),
        )

    with time_block("build MS MARCO triplet eval dataset"):
        eval_dataset = MSMarcoTripletDataset(
            split=data_cfg["eval_split"],
            hf_dataset_name=data_cfg["hf_dataset_name"],
            hf_config_name=data_cfg["hf_config_name"],
            max_samples=int(data_cfg.get("max_eval_samples", 5000)),
        )

    print(
        f"[train_biencoder] train_size={len(train_dataset)}, "
        f"eval_size={len(eval_dataset)}"
    )

    # Tokenizer + collator
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    collator = BiEncoderCollator(
        tokenizer=tokenizer,
        max_query_length=int(data_cfg.get("max_query_length", 32)),
        max_passage_length=int(data_cfg.get("max_passage_length", 128)),
    )

    # DataLoaders
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 2))
    pin_memory = device.type == "cuda"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )

    # Model
    projection_dim = train_cfg.get("projection_dim", None)
    if projection_dim is None:
        projection_dim = cfg["model"].get("projection_dim", None)

    model = BiEncoderModel(
        model_name=model_cfg["name"],
        pooling=model_cfg.get("pooling", "cls"),
        projection_dim=projection_dim,
    )
    model.to(device)

    # Optimizer & scheduler
    num_epochs = int(train_cfg["num_epochs"])
    total_training_steps = len(train_dataloader) * num_epochs

    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        total_training_steps=total_training_steps,
        warmup_ratio=float(train_cfg["warmup_ratio"]),
    )

    trainer = BiEncoderTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        output_dir=train_cfg["output_dir"],
        num_epochs=num_epochs,
        temperature=float(train_cfg["temperature"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        log_every_steps=int(train_cfg.get("log_every_steps", 100)),
        save_best=bool(train_cfg.get("save_best", True)),
    )

    trainer.train()


if __name__ == "__main__":
    main()

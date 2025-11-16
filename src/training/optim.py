# src/training/optim.py
"""
Optimizer and scheduler utilities for training models.

This module defines helper functions to construct an AdamW optimizer
and a linear warmup scheduler for Transformer-based models.
"""

from __future__ import annotations

from typing import Tuple

from torch.optim import Optimizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    total_training_steps: int,
    warmup_ratio: float,
) -> Tuple[Optimizer, object]:
    """
    Create an AdamW optimizer and a linear warmup scheduler.

    Args:
        model:
            PyTorch model whose parameters will be optimized.
        learning_rate:
            Base learning rate for AdamW.
        weight_decay:
            Weight decay factor for AdamW.
        total_training_steps:
            Total number of training steps (batches * epochs).
        warmup_ratio:
            Fraction of total steps used for learning rate warmup.

    Returns:
        Tuple of (optimizer, scheduler) where scheduler is a
        transformers.get_linear_schedule_with_warmup instance.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(param_groups, lr=learning_rate)

    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    return optimizer, scheduler

# src/training/losses.py
"""
Loss functions for training neural retrieval models.

This module currently provides a contrastive InfoNCE-style loss
for bi-encoder training using in-batch negatives.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def contrastive_loss(
    query_embeddings: torch.Tensor,
    passage_embeddings: torch.Tensor,
    temperature: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an InfoNCE contrastive loss with in-batch negatives.

    Args:
        query_embeddings:
            Tensor of query embeddings (batch_size, dim).
        passage_embeddings:
            Tensor of passage embeddings (batch_size, dim).
        temperature:
            Temperature scaling factor applied to similarities.

    Returns:
        Tuple containing:
        - loss: Scalar contrastive loss tensor.
        - logits: Similarity logits matrix of shape (batch_size, batch_size).
    """
    query_embeddings = nn.functional.normalize(query_embeddings, dim=-1)
    passage_embeddings = nn.functional.normalize(passage_embeddings, dim=-1)

    logits = torch.matmul(query_embeddings, passage_embeddings.t())
    logits = logits / temperature

    batch_size = query_embeddings.size(0)
    targets = torch.arange(batch_size, dtype=torch.long, device=logits.device)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    return loss, logits

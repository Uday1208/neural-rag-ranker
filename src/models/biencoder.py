# src/models/biencoder.py
"""
Bi-encoder neural ranker model for dense retrieval.

This module defines a BiEncoderModel that wraps a Hugging Face
Transformer backbone and produces vector representations for
queries and passages using configurable pooling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class BiEncoderModel(nn.Module):
    """
    Bi-encoder model that encodes queries and passages independently.

    The underlying encoder weights are shared between the query and
    passage towers and use the same Transformer backbone.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        projection_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize the BiEncoderModel from a Hugging Face checkpoint.

        Args:
            model_name:
                Name or path of the Hugging Face model to load.
            pooling:
                Pooling strategy to apply ("cls" or "mean").
            projection_dim:
                Optional output dimension for a linear projection layer;
                if None, the encoder hidden size is used as-is.
        """
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        self.pooling = pooling.lower()
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("Pooling must be one of: 'cls', 'mean'.")

        hidden_size = self.config.hidden_size
        if projection_dim is not None:
            self.projection = nn.Linear(hidden_size, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = hidden_size

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply pooling over the encoder outputs.

        Args:
            last_hidden_state:
                Tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask:
                Tensor of shape (batch_size, seq_len) indicating valid tokens.

        Returns:
            Pooled embeddings of shape (batch_size, hidden_size or projection_dim).
        """
        if self.pooling == "cls":
            pooled = last_hidden_state[:, 0]
        else:
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
            lengths = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = sum_embeddings / lengths

        if self.projection is not None:
            pooled = self.projection(pooled)

        return pooled

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode input token IDs and return pooled embeddings.

        Args:
            input_ids:
                Tensor of token IDs (batch_size, seq_len).
            attention_mask:
                Tensor of attention mask values (batch_size, seq_len).

        Returns:
            Tensor of pooled embeddings (batch_size, output_dim).
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        pooled = self._pool(last_hidden_state, attention_mask)
        return pooled

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode queries and passages and return their embeddings.

        Args:
            query_input_ids:
                Tensor of token IDs for queries (batch_size, seq_len_q).
            query_attention_mask:
                Attention mask for queries (batch_size, seq_len_q).
            passage_input_ids:
                Tensor of token IDs for passages (batch_size, seq_len_p).
            passage_attention_mask:
                Attention mask for passages (batch_size, seq_len_p).

        Returns:
            Tuple of (query_embeddings, passage_embeddings) tensors.
        """
        query_emb = self.encode(query_input_ids, query_attention_mask)
        passage_emb = self.encode(passage_input_ids, passage_attention_mask)
        return query_emb, passage_emb

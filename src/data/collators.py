# src/data/collators.py
"""
Batch collation utilities for converting raw dataset examples into
tokenized tensors suitable for bi-encoder and cross-encoder models.

This module currently provides a BiEncoderCollator that uses a
Hugging Face tokenizer to encode query and passage texts into
PyTorch tensors.
"""

from __future__ import annotations

from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase

from src.data.msmarco_dataset import MSMarcoTripletExample


class BiEncoderCollator:
    """
    Collate function for bi-encoder training batches on MS MARCO.

    This collator tokenizes queries and positive passages into
    separate input tensors for the bi-encoder model.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_query_length: int = 32,
        max_passage_length: int = 128,
    ) -> None:
        """
        Initialize the collator with a tokenizer and sequence lengths.

        Args:
            tokenizer:
                Hugging Face tokenizer used for encoding text.
            max_query_length:
                Maximum number of tokens for the query sequence.
            max_passage_length:
                Maximum number of tokens for the passage sequence.
        """
        self._tokenizer = tokenizer
        self._max_query_length = max_query_length
        self._max_passage_length = max_passage_length

    def __call__(
        self,
        batch: List[MSMarcoTripletExample],
    ) -> Dict[str, Any]:
        """
        Convert a batch of MS MARCO triplet examples into tokenized tensors.

        Args:
            batch:
                List of MSMarcoTripletExample objects.

        Returns:
            Dictionary with query and passage tokenization outputs suitable
            for direct input into a bi-encoder model.
        """
        queries = [ex.query for ex in batch]
        passages = [ex.positive_passage for ex in batch]

        query_enc = self._tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self._max_query_length,
            return_tensors="pt",
        )
        passage_enc = self._tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=self._max_passage_length,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "passage_input_ids": passage_enc["input_ids"],
            "passage_attention_mask": passage_enc["attention_mask"],
        }

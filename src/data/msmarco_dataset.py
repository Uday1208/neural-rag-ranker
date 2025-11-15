# src/data/msmarco_dataset.py
"""
PyTorch Dataset wrappers for the MS MARCO Passage Ranking dataset.

This module provides a triplet-style Dataset that reads from the
`microsoft/ms_marco` v1.1 Hugging Face dataset and yields
(query, positive_passage, negative_passage) text triples suitable
for contrastive training of bi-encoder rankers.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class MSMarcoTripletExample:
    """
    Simple data container for a single MS MARCO triplet example.

    Attributes:
        query: The input query text as a string.
        positive_passage: A passage known to be relevant to the query.
        negative_passage: A passage assumed to be non-relevant for the query.
    """

    query: str
    positive_passage: str
    negative_passage: str


def _has_positive_and_negative(example: Dict[str, Any]) -> bool:
    """
    Check whether a dataset row contains at least one positive and one negative.

    Args:
        example:
            A single row from the MS MARCO dataset, including a "passages"
            field with "is_selected" flags.

    Returns:
        True if the example has both positive and negative passages.
    """
    flags = example["passages"]["is_selected"]
    has_pos = any(bool(flag) for flag in flags)
    has_neg = any(not bool(flag) for flag in flags)
    return has_pos and has_neg


class MSMarcoTripletDataset(Dataset):
    """
    PyTorch Dataset producing (query, positive, negative) text triples.

    This dataset wraps `microsoft/ms_marco` v1.1 and, for each index, samples
    one positive and one negative passage associated with the query.
    """

    def __init__(
        self,
        split: str,
        hf_dataset_name: str = "microsoft/ms_marco",
        hf_config_name: str = "v1.1",
        max_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the MS MARCO triplet dataset from Hugging Face Datasets.

        Args:
            split:
                Dataset split name to load (e.g., "train" or "validation").
            hf_dataset_name:
                Name of the Hugging Face dataset to load.
            hf_config_name:
                Configuration name within the Hugging Face dataset.
            max_samples:
                Optional maximum number of rows to retain after filtering; if
                None, all available rows for the split are used.
            seed:
                Seed value used for per-item random sampling of passages.
            cache_dir:
                Optional directory for Hugging Face dataset caching.

        Raises:
            ValueError: If no valid examples remain after filtering.
        """
        super().__init__()
        self._rng = random.Random(seed)
        self._split = split
        self._hf_dataset_name = hf_dataset_name
        self._hf_config_name = hf_config_name

        logger.info(
            "Loading MS MARCO dataset '%s' (config='%s', split='%s')",
            hf_dataset_name,
            hf_config_name,
            split,
        )

        hf_ds: HFDataset = load_dataset(
            hf_dataset_name,
            hf_config_name,
            split=split,
            cache_dir=cache_dir,
        )

        if max_samples is not None:
            max_samples = min(max_samples, len(hf_ds))
            hf_ds = hf_ds.select(range(max_samples))
            logger.info(
                "Selected first %d samples from split '%s'", max_samples, split
            )

        logger.info("Filtering examples without both positive and negative passages")

        # The filter function must be picklable; defining it at module level satisfies this.
        hf_ds = hf_ds.filter(_has_positive_and_negative)

        if len(hf_ds) == 0:
            raise ValueError(
                "No valid MS MARCO examples left after filtering for positives and "
                "negatives. Please check your dataset configuration."
            )

        logger.info("Final dataset size after filtering: %d", len(hf_ds))

        self._dataset: HFDataset = hf_ds

    def __len__(self) -> int:
        """
        Return the number of examples in the dataset.

        Returns:
            Number of triplet-capable rows in the underlying dataset.
        """
        return len(self._dataset)

    def __getitem__(self, idx: int) -> MSMarcoTripletExample:
        """
        Retrieve a single triplet example by index.

        Args:
            idx:
                Integer index into the filtered MS MARCO dataset.

        Returns:
            An MSMarcoTripletExample containing query, positive, and negative text.
        """
        row: Dict[str, Any] = self._dataset[int(idx)]

        query_text: str = row["query"]
        passages: List[str] = row["passages"]["passage_text"]
        flags: List[Any] = row["passages"]["is_selected"]

        positive_indices: List[int] = [
            i for i, flag in enumerate(flags) if bool(flag)
        ]
        negative_indices: List[int] = [
            i for i, flag in enumerate(flags) if not bool(flag)
        ]

        if not positive_indices or not negative_indices:
            # This should not happen due to the filter, but we guard anyway.
            raise RuntimeError(
                "Encountered MS MARCO row without both positives and negatives "
                f"at index {idx}; consider re-running dataset filtering."
            )

        pos_idx = self._rng.choice(positive_indices)
        neg_idx = self._rng.choice(negative_indices)

        positive_passage = passages[pos_idx]
        negative_passage = passages[neg_idx]

        return MSMarcoTripletExample(
            query=query_text,
            positive_passage=positive_passage,
            negative_passage=negative_passage,
        )

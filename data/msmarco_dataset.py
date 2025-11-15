from dataclasses import dataclass  # if already imported, it is safe to keep

from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset, load_dataset

from src.utils.timing import time_block


@dataclass
class MSMarcoRankingExample:
    """
    Data container for a single ranking-style MS MARCO example.

    Attributes:
        query_id: Unique string identifier for the query.
        query: Query text string.
        passages: List of candidate passage texts for this query.
        relevance_labels: List of 0/1 relevance labels for each passage.
    """

    query_id: str
    query: str
    passages: List[str]
    relevance_labels: List[int]


def _has_positive(example: Dict[str, Any]) -> bool:
    """
    Check whether a dataset row contains at least one positive passage.

    Args:
        example:
            A single row from the MS MARCO dataset, including "passages"
            and "is_selected" relevance flags.

    Returns:
        True if the example has at least one positive passage.
    """
    flags = example["passages"]["is_selected"]
    return any(bool(flag) for flag in flags)


class MSMarcoRankingDataset(Dataset):
    """
    PyTorch Dataset providing per-query ranking candidates for MS MARCO.

    Each item contains the query ID, query text, a list of candidate
    passages, and binary relevance labels indicating which passages are
    judged relevant for the query.
    """

    def __init__(
        self,
        split: str,
        hf_dataset_name: str = "microsoft/ms_marco",
        hf_config_name: str = "v1.1",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the MS MARCO ranking dataset from Hugging Face Datasets.

        Args:
            split:
                Dataset split name to load (e.g., "train" or "validation").
            hf_dataset_name:
                Name of the Hugging Face dataset to load.
            hf_config_name:
                Configuration name within the Hugging Face dataset.
            max_samples:
                Optional maximum number of rows to keep after selection;
                if None, all rows for the split are used.
            cache_dir:
                Optional directory used by Hugging Face for caching.

        Raises:
            ValueError: If no examples remain after filtering positives.
        """
        super().__init__()

        with time_block(f"load+filter MS MARCO ranking split='{split}'"):
            hf_ds: HFDataset = load_dataset(
                hf_dataset_name,
                hf_config_name,
                split=split,
                cache_dir=cache_dir,
            )

            if max_samples is not None:
                max_samples = min(max_samples, len(hf_ds))
                hf_ds = hf_ds.select(range(max_samples))

            hf_ds = hf_ds.filter(_has_positive)

        if len(hf_ds) == 0:
            raise ValueError(
                "No MS MARCO ranking examples left after filtering for positives."
            )

        self._dataset: HFDataset = hf_ds

    def __len__(self) -> int:
        """
        Return the number of queries in the ranking dataset.

        Returns:
            Integer number of MS MARCO queries with at least one positive.
        """
        return len(self._dataset)

    def __getitem__(self, idx: int) -> MSMarcoRankingExample:
        """
        Retrieve a single ranking example by index.

        Args:
            idx:
                Integer index into the filtered MS MARCO dataset.

        Returns:
            MSMarcoRankingExample with query, passages, and relevance labels.
        """
        row: Dict[str, Any] = self._dataset[int(idx)]

        query_id: str = str(row.get("query_id"))
        query_text: str = row["query"]
        passages: List[str] = list(row["passages"]["passage_text"])
        flags = row["passages"]["is_selected"]
        relevance_labels: List[int] = [1 if bool(f) else 0 for f in flags]

        return MSMarcoRankingExample(
            query_id=query_id,
            query=query_text,
            passages=passages,
            relevance_labels=relevance_labels,
        )

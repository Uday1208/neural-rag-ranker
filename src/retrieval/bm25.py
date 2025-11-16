# src/retrieval/bm25.py
"""
BM25 retrieval baseline implementation for ranking candidates.

This module wraps the rank_bm25.BM25Okapi implementation to provide a
simple BM25Retriever class that can score and rank candidate passages
for a given query. It is intended for baseline evaluation against
neural rankers within this project.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi


def default_tokenizer(text: str) -> List[str]:
    """
    Tokenize a text string into BM25 tokens.

    Args:
        text:
            Input text string to be tokenized.

    Returns:
        List of lowercase tokens split on whitespace.
    """
    return text.lower().split()


class BM25Retriever:
    """
    Lightweight BM25-based retriever for ranking text passages.

    The retriever is constructed with a fixed set of documents and then
    can rank those documents for arbitrary query strings using BM25Okapi.
    """

    def __init__(
        self,
        documents: Sequence[str],
        doc_ids: Optional[Sequence[str]] = None,
        tokenizer: Callable[[str], List[str]] = default_tokenizer,
    ) -> None:
        """
        Initialize the BM25 retriever with a document corpus.

        Args:
            documents:
                Sequence of document text strings to index.
            doc_ids:
                Optional sequence of document IDs aligned with `documents`;
                if None, integer string IDs ('0', '1', ...) are generated.
            tokenizer:
                Callable that converts input strings into token lists.
        """
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        if len(doc_ids) != len(documents):
            raise ValueError("Length of doc_ids must match length of documents.")

        self._tokenizer = tokenizer
        self._doc_ids: List[str] = list(doc_ids)
        tokenized_docs: List[List[str]] = [tokenizer(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank the indexed documents for a given query string.

        Args:
            query:
                Query text string to be ranked against the corpus.
            top_k:
                Maximum number of top documents to return.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        query_tokens = self._tokenizer(query)
        scores = self._bm25.get_scores(query_tokens)

        doc_scores: List[Tuple[str, float]] = list(
            zip(self._doc_ids, [float(s) for s in scores])
        )
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

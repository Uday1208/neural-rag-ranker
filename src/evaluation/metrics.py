# src/evaluation/metrics.py
"""
Information retrieval metrics for ranking evaluation.

This module implements common IR metrics such as MRR@K, Recall@K,
Precision@K, and nDCG@K for query–document ranking runs. The metrics
operate over TREC-style qrels and run dictionaries so they can be
reused across BM25, bi-encoder, and cross-encoder rankers.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence


Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def _dcg(relevances: Iterable[int]) -> float:
    """
    Compute DCG (Discounted Cumulative Gain) from a list of relevances.

    Args:
        relevances:
            Iterable of integer relevance scores in ranked order.

    Returns:
        DCG value as a float.
    """
    dcg = 0.0
    for idx, rel in enumerate(relevances):
        if rel <= 0:
            continue
        dcg += (2.0 ** float(rel) - 1.0) / math.log2(float(idx + 2))
    return dcg


def compute_mrr_at_k(qrels: Qrels, run: Run, k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) at cutoff K.

    Args:
        qrels:
            Mapping from query ID to a mapping of doc ID to relevance score.
        run:
            Mapping from query ID to an ordered list of retrieved doc IDs.
        k:
            Rank cutoff at which to truncate the ranked list.

    Returns:
        Mean reciprocal rank across all queries with at least one relevant doc.
    """
    total_rr = 0.0
    valid_queries = 0

    for qid, doc_rels in qrels.items():
        relevant_docs = {doc_id for doc_id, rel in doc_rels.items() if rel > 0}
        if not relevant_docs:
            continue

        ranking = run.get(qid, [])
        rank_found = None

        for rank, doc_id in enumerate(ranking[:k]):
            if doc_id in relevant_docs:
                rank_found = rank + 1
                break

        if rank_found is not None:
            total_rr += 1.0 / float(rank_found)

        valid_queries += 1

    if valid_queries == 0:
        return 0.0

    return total_rr / float(valid_queries)


def compute_precision_at_k(qrels: Qrels, run: Run, k: int = 10) -> float:
    """
    Compute Precision@K for a ranking run.

    Args:
        qrels:
            Mapping from query ID to relevance judgments.
        run:
            Mapping from query ID to ranked doc ID lists.
        k:
            Number of top-ranked documents to consider.

    Returns:
        Mean Precision@K across all queries with at least one relevant doc.
    """
    total_precision = 0.0
    valid_queries = 0

    for qid, doc_rels in qrels.items():
        relevant_docs = {doc_id for doc_id, rel in doc_rels.items() if rel > 0}
        if not relevant_docs:
            continue

        ranking = run.get(qid, [])
        top_docs = ranking[:k]

        num_relevant_in_top_k = sum(1 for doc_id in top_docs if doc_id in relevant_docs)
        precision = num_relevant_in_top_k / float(k)

        total_precision += precision
        valid_queries += 1

    if valid_queries == 0:
        return 0.0

    return total_precision / float(valid_queries)


def compute_recall_at_k(qrels: Qrels, run: Run, k: int = 10) -> float:
    """
    Compute Recall@K for a ranking run.

    Args:
        qrels:
            Mapping from query ID to relevance judgments.
        run:
            Mapping from query ID to ranked doc ID lists.
        k:
            Number of top-ranked documents to consider.

    Returns:
        Mean Recall@K across all queries with at least one relevant doc.
    """
    total_recall = 0.0
    valid_queries = 0

    for qid, doc_rels in qrels.items():
        relevant_docs = {doc_id for doc_id, rel in doc_rels.items() if rel > 0}
        if not relevant_docs:
            continue

        ranking = run.get(qid, [])
        top_docs = set(ranking[:k])

        num_relevant_in_top_k = len(relevant_docs & top_docs)
        recall = num_relevant_in_top_k / float(len(relevant_docs))

        total_recall += recall
        valid_queries += 1

    if valid_queries == 0:
        return 0.0

    return total_recall / float(valid_queries)


def compute_ndcg_at_k(qrels: Qrels, run: Run, k: int = 10) -> float:
    """
    Compute nDCG@K (normalized Discounted Cumulative Gain).

    Args:
        qrels:
            Mapping from query ID to relevance judgments.
        run:
            Mapping from query ID to ranked doc ID lists.
        k:
            Rank cutoff for DCG and IDCG calculation.

    Returns:
        Mean nDCG@K across all queries with at least one relevant doc.
    """
    total_ndcg = 0.0
    valid_queries = 0

    for qid, doc_rels in qrels.items():
        if not doc_rels:
            continue

        ranking = run.get(qid, [])
        ranked_relevances: List[int] = [
            int(doc_rels.get(doc_id, 0)) for doc_id in ranking[:k]
        ]

        dcg = _dcg(ranked_relevances)

        ideal_relevances = sorted(
            (int(rel) for rel in doc_rels.values() if rel > 0),
            reverse=True,
        )[:k]

        if not ideal_relevances:
            continue

        idcg = _dcg(ideal_relevances)
        if idcg <= 0.0:
            continue

        total_ndcg += dcg / idcg
        valid_queries += 1

    if valid_queries == 0:
        return 0.0

    return total_ndcg / float(valid_queries)


def compute_metrics_at_ks(
    qrels: Qrels,
    run: Run,
    ks: Iterable[int],
) -> Dict[str, float]:
    """
    Compute a dictionary of IR metrics at multiple K values.

    Args:
        qrels:
            Mapping from query ID to relevance judgments.
        run:
            Mapping from query ID to ranked doc ID lists.
        ks:
            Iterable of integer K values at which to compute metrics.

    Returns:
        Dictionary mapping metric names such as "MRR@10" to float values.
    """
    results: Dict[str, float] = {}

    for k in ks:
        results[f"MRR@{k}"] = compute_mrr_at_k(qrels, run, k=k)
        results[f"Precision@{k}"] = compute_precision_at_k(qrels, run, k=k)
        results[f"Recall@{k}"] = compute_recall_at_k(qrels, run, k=k)
        results[f"nDCG@{k}"] = compute_ndcg_at_k(qrels, run, k=k)

    return results

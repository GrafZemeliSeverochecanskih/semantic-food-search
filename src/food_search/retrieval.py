"""Hybrid retrieval: BM25 + dense cosine, fused with Reciprocal Rank Fusion.

RRF operates on ranks rather than raw scores, so the incompatible scales of
BM25 and cosine similarity never need calibration:

    score(d) = sum over channels of 1 / (rrf_k + rank_channel(d))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .data import Item, tokenize
from .embedders import Embedder


@dataclass
class SearchResult:
    item: Item
    score: float
    channels: dict[str, int] # channel name -> 1-based rank in that channel


class BM25Index:
    def __init__(self, items: list[Item]):
        self.corpus_tokens = [tokenize(item.document()) for item in items]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def rank(self, query: str, top_n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_n]
        # don't pad the pool with zero-score non-matches
        return [int(i) for i in order if scores[i] > 0]


class DenseIndex:
    """Exact cosine search over normalized vectors - at 5k items a matrix
    product is faster and simpler than an ANN index."""

    def __init__(self, matrix: np.ndarray, embedder: Embedder):
        self.matrix = matrix
        self.embedder = embedder

    def rank(self, query: str, top_n: int) -> list[int]:
        q = self.embedder.embed([query], kind="query")[0]
        sims = self.matrix @ q # vectors are L2-normalized, so dot = cosine
        return [int(i) for i in np.argsort(sims)[::-1][:top_n]]


def rrf_fuse(
    rankings: dict[str, list[int]],
    k: int = 60,
) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse per-channel rankings; returns (doc_idx, rrf_score, per-channel ranks)."""
    scores: dict[int, float] = {}
    provenance: dict[int, dict[str, int]] = {} # per-channel ranks, for explainability
    for channel, ranking in rankings.items():
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
            provenance.setdefault(doc_idx, {})[channel] = rank
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(idx, score, provenance[idx]) for idx, score in fused]


class HybridSearcher:
    def __init__(
        self,
        items: list[Item],
        bm25: BM25Index,
        dense: DenseIndex,
        cfg,
    ):
        self.items = items
        self.bm25 = bm25
        self.dense = dense
        self.cfg = cfg

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        top_k = top_k or self.cfg.top_k
        depth = self.cfg.candidates_per_channel
        rankings: dict[str, list[int]] = {}
        if mode in ("hybrid", "bm25"):
            rankings["bm25"] = self.bm25.rank(query, depth)
        if mode in ("hybrid", "dense"):
            rankings["dense"] = self.dense.rank(query, depth)
        # single-channel modes go through fusion too: with one ranking,
        # 1/(k+rank) is monotonic so the channel order is preserved
        fused = rrf_fuse(rankings, k=self.cfg.rrf_k)
        return [
            SearchResult(item=self.items[idx], score=score, channels=chans)
            for idx, score, chans in fused[:top_k]
        ]

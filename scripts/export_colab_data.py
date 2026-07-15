"""Package everything the Colab training scripts need into one JSON file.

Produces artifacts/colab_data.json containing:
  * train_pairs   - (query, passage, grade) for the 60 train-split queries
  * test_queries  - for each of the 40 held-out queries: the hybrid top-20
                    candidates (passage + known grade) and the full pool
                    grades (for the nDCG ideal)
  * baselines     - reference numbers to print next to new results

Runs fully offline: corpus and query embeddings are read from the local
caches, judgments from the store. No API calls.

The export contains only the item snippets that appear in pools/candidates,
not the full catalog. It is still derived from confidential data - upload it
to your private Colab session only, and delete it from Colab afterwards.

Usage:
    python scripts/export_colab_data.py
"""

from __future__ import annotations

import json

import _bootstrap # noqa: F401

from food_search.config import Config
from food_search.data import load_items
from food_search.embedders import QueryCachedEmbedder, embed_corpus_cached, make_embedder
from food_search.retrieval import BM25Index, DenseIndex, HybridSearcher

SEP = "\x1e"

BASELINES = {
    "hybrid (no rerank)": {"nDCG@10": 0.550, "MRR@10": 0.774, "P@5": 0.525},
    "zero-shot MiniLM CE 118M": {"nDCG@10": 0.475, "MRR@10": 0.581, "P@5": 0.475},
    "zero-shot bge-reranker 568M": {"nDCG@10": 0.595, "MRR@10": 0.749, "P@5": 0.525},
    "distilled MiniLM CE 118M": {"nDCG@10": 0.595, "MRR@10": 0.762, "P@5": 0.550},
    "LLM rerank gpt-4.1-mini": {"nDCG@10": 0.806, "MRR@10": 0.891, "P@5": 0.715},
}


def passage_text(item) -> str:
    # Same passage format the rerankers use, so Colab models train on it too.
    parts = [item.name]
    if item.description:
        parts.append(item.description[:200])
    if item.category:
        parts.append(item.category)
    return ". ".join(parts)


def main() -> None:
    cfg = Config(embedder="openai")
    items = load_items(cfg.items_csv)
    items_by_id = {it.item_id: it for it in items}

    # Regroup the flat (query \x1e item_id) -> grade store into per-query dicts.
    judgments = json.loads((cfg.artifacts_dir / "judgments.json").read_text(encoding="utf-8"))
    graded: dict[str, dict[str, int]] = {}
    for key, grade in judgments.items():
        query, item_id = key.split(SEP)
        graded.setdefault(query, {})[item_id] = grade

    split = json.loads((cfg.artifacts_dir / "distill_split.json").read_text(encoding="utf-8"))
    train_queries, test_queries = split["train"], split["test"]

    train_pairs = [
        {"query": q, "passage": passage_text(items_by_id[iid]), "grade": g}
        for q in train_queries
        for iid, g in graded.get(q, {}).items()
        if iid in items_by_id
    ]

    embedder = QueryCachedEmbedder(make_embedder("openai", cfg), cfg.artifacts_dir)
    matrix = embed_corpus_cached(embedder, [it.document() for it in items], cfg.artifacts_dir)
    searcher = HybridSearcher(items, BM25Index(items), DenseIndex(matrix, embedder), cfg)

    test_blocks = []
    unjudged = 0
    for query in test_queries:
        candidates = []
        for res in searcher.search(query, top_k=cfg.rerank_pool, mode="hybrid"):
            grade = graded.get(query, {}).get(res.item.item_id)
            unjudged += grade is None
            candidates.append(
                {"item_id": res.item.item_id, "passage": passage_text(res.item), "grade": grade}
            )
        test_blocks.append(
            {"query": query, "candidates": candidates, "pool_grades": sorted(graded.get(query, {}).values(), reverse=True)}
        )

    out = cfg.artifacts_dir / "colab_data.json"
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "train_queries": len(train_queries),
                    "test_queries": len(test_queries),
                    "train_pairs": len(train_pairs),
                    "unjudged_test_candidates": unjudged,
                    "note": "unjudged candidates score as grade 0 in Colab metrics (conservative); "
                    "for exact numbers, bring the results CSV back and run scripts/evaluate.py",
                },
                "baselines": BASELINES,
                "train_pairs": train_pairs,
                "test_queries": test_blocks,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"{len(train_pairs)} train pairs, {len(test_blocks)} test queries "
          f"({unjudged} of {len(test_blocks) * cfg.rerank_pool} test candidates unjudged)")
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

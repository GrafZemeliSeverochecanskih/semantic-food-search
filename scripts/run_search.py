"""Run the 100 evaluation queries through one or more system variants.

Usage:
    python scripts/run_search.py --embedder local --systems bm25 dense hybrid
    python scripts/run_search.py --embedder openai --systems hybrid hybrid_rerank

Writes outputs/results_<system>_<embedder>.csv with the top-k per query.
"""

from __future__ import annotations

import argparse
import csv
import time

import _bootstrap # noqa: F401

from food_search.config import Config
from food_search.data import load_items, load_queries
from food_search.embedders import QueryCachedEmbedder, embed_corpus_cached, make_embedder
from food_search.retrieval import BM25Index, DenseIndex, HybridSearcher

SYSTEMS = ("bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_ce")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=("openai", "local"), default="openai")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=["hybrid"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ce-model", default="minilm", help="cross-encoder alias or HF id (for hybrid_ce)")
    args = parser.parse_args()

    cfg = Config(embedder=args.embedder, top_k=args.top_k)
    items = load_items(cfg.items_csv)
    queries = load_queries(cfg.queries_csv)
    print(f"Loaded {len(items)} items, {len(queries)} queries")

    embedder = QueryCachedEmbedder(make_embedder(args.embedder, cfg), cfg.artifacts_dir)
    t0 = time.time()
    matrix = embed_corpus_cached(embedder, [it.document() for it in items], cfg.artifacts_dir)
    print(f"Corpus embeddings ready in {time.time() - t0:.1f}s ({embedder.name})")

    searcher = HybridSearcher(items, BM25Index(items), DenseIndex(matrix, embedder), cfg)
    # Rerankers are built lazily: no API client / model download unless requested.
    rerankers = {}
    if "hybrid_rerank" in args.systems:
        from food_search.rerank import LLMReranker

        rerankers["hybrid_rerank"] = LLMReranker(cfg.rerank_model)
    if "hybrid_ce" in args.systems:
        from food_search.rerank import CrossEncoderReranker

        rerankers["hybrid_ce"] = CrossEncoderReranker(args.ce_model)

    embedder_tag = args.embedder
    for system in args.systems:
        tag = f"{system}-{args.ce_model}" if system == "hybrid_ce" else system
        out_path = cfg.outputs_dir / f"results_{tag}_{embedder_tag}.csv"
        t0 = time.time()
        with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["query", "rank", "itemId", "name", "description", "category", "price", "score", "channels", "image_url"]
            )
            for query in queries:
                if system in rerankers:
                    # deeper hybrid pool -> rerank -> cut to top-k
                    candidates = searcher.search(query, top_k=cfg.rerank_pool, mode="hybrid")
                    results = rerankers[system].rerank(query, candidates)[: cfg.top_k]
                else:
                    results = searcher.search(query, top_k=cfg.top_k, mode=system.split("_")[0])
                for rank, res in enumerate(results, start=1):
                    writer.writerow(
                        [
                            query,
                            rank,
                            res.item.item_id,
                            res.item.name,
                            res.item.description,
                            res.item.category,
                            res.item.price,
                            f"{res.score:.5f}",
                            "|".join(f"{c}:{r}" for c, r in res.channels.items()),
                            res.item.image_url or "",
                        ]
                    )
        print(f"[{system}] {len(queries)} queries in {time.time() - t0:.1f}s -> {out_path.name}")


if __name__ == "__main__":
    main()

"""Interactive demo: type Portuguese food queries, get ranked items.

Default mode `best` runs the winning system end to end: BM25 + three dense
views (PT query, English translation, HyDE hypothetical item) fused with RRF,
then a listwise LLM rerank. The single-channel modes (bm25/dense/hybrid) are
kept for comparison. `best` needs the OpenAI API (views + rerank); with
`--embedder local` it falls back to `hybrid`.

Usage:
    python scripts/demo.py                 # best system (needs OPENAI_API_KEY)
    python scripts/demo.py --mode hybrid   # retrieval only, no API rerank
    python scripts/demo.py --embedder local --mode hybrid   # fully offline
"""

from __future__ import annotations

import argparse
import json

import _bootstrap # noqa: F401
import numpy as np

from food_search.config import Config
from food_search.data import load_items
from food_search.embedders import QueryCachedEmbedder, embed_corpus_cached, make_embedder
from food_search.retrieval import BM25Index, DenseIndex, HybridSearcher, SearchResult, rrf_fuse

VIEW_PROMPT = (
    "You expand a Portuguese food-delivery search query into two views. "
    "Reply with JSON only:\n"
    '{"en": "<natural English translation of the query>", '
    '"hyde": "<in Portuguese: name and one-sentence menu description of a '
    'plausible catalog item that perfectly satisfies the query>"}'
)


class BestSystem:
    """Multi-view fusion + listwise LLM rerank, with a cached view store."""

    def __init__(
        self,
        items,
        matrix,
        embedder,
        bm25,
        cfg,
    ):
        from openai import OpenAI

        from food_search.rerank import LLMReranker

        self.items, self.matrix, self.embedder, self.bm25, self.cfg = items, matrix, embedder, bm25, cfg
        self.client = OpenAI()
        self.reranker = LLMReranker(cfg.rerank_model)
        self.views_path = cfg.artifacts_dir / "query_views.json"
        self.views = json.loads(self.views_path.read_text(encoding="utf-8")) if self.views_path.exists() else {}

    def _get_views(self, query: str) -> dict:
        if query in self.views:
            return self.views[query] # cache hit for the 100 eval queries
        response = self.client.chat.completions.create(
            model=self.cfg.rerank_model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": VIEW_PROMPT}, {"role": "user", "content": query}],
        )
        views = json.loads(response.choices[0].message.content)
        self.views[query] = {"en": str(views.get("en", query)), "hyde": str(views.get("hyde", query))}
        self.views_path.write_text(json.dumps(self.views, ensure_ascii=False, indent=0), encoding="utf-8")
        return self.views[query]

    def _dense(self, text: str, kind: str, top_n: int) -> list[int]:
        sims = self.matrix @ self.embedder.embed([text], kind=kind)[0]
        return [int(j) for j in np.argsort(sims)[::-1][:top_n]]

    def search(self, query: str) -> list[SearchResult]:
        view = self._get_views(query)
        depth = self.cfg.candidates_per_channel
        rankings = {
            "bm25": self.bm25.rank(query, depth),
            "pt": self._dense(query, "query", depth),
            "en": self._dense(view["en"], "query", depth),
            "hyde": self._dense(view["hyde"], "passage", depth), # hypothetical item is document-like
        }
        fused = rrf_fuse(rankings, k=self.cfg.rrf_k)
        pool = [SearchResult(self.items[j], s, ch) for j, s, ch in fused[: self.cfg.rerank_pool]]
        return self.reranker.rerank(query, pool)[: self.cfg.top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=("openai", "local"), default="openai")
    parser.add_argument("--mode", choices=("best", "hybrid", "bm25", "dense"), default="best")
    args = parser.parse_args()

    cfg = Config(embedder=args.embedder)
    items = load_items(cfg.items_csv)
    embedder = QueryCachedEmbedder(make_embedder(args.embedder, cfg), cfg.artifacts_dir)
    matrix = embed_corpus_cached(embedder, [it.document() for it in items], cfg.artifacts_dir)
    bm25 = BM25Index(items)

    mode = args.mode
    if mode == "best" and args.embedder == "local":
        print("note: 'best' mode needs the OpenAI API; falling back to 'hybrid'.")
        mode = "hybrid"

    best = BestSystem(items, matrix, embedder, bm25, cfg) if mode == "best" else None
    searcher = HybridSearcher(items, bm25, DenseIndex(matrix, embedder), cfg)
    print(f"Ready ({len(items)} items, {embedder.name}, mode={mode}). Empty line to quit.\n")

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        results = best.search(query) if best else searcher.search(query, mode=mode)
        for rank, res in enumerate(results, start=1):
            # channels shows retrieval provenance, e.g. "bm25#5, hyde#1" = rank per channel
            channels = ", ".join(f"{c}#{r}" for c, r in res.channels.items())
            price = f"R${res.item.price:.2f}" if res.item.price is not None else "-"
            print(f"{rank:2d}. {res.item.name}  [{res.item.category}] {price}  ({channels})")
            if res.item.description:
                print(f"      {res.item.description[:110]}")
        print()


if __name__ == "__main__":
    main()

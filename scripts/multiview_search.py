"""Multi-view dense retrieval: PT query + EN translation + HyDE passage.

Rationale: multilingual embedding models are trained on English-heavy data,
so an English translation of the query can be better placed in embedding
space than the Portuguese original; and for stylistic queries ("comida de
rua do nordeste") a hypothetical catalog item written by an LLM (HyDE)
turns the fragile query->document match into a better-conditioned
document->document match. Both views cost one small LLM call per query,
cached in artifacts/query_views.json.

Systems produced (standard results CSV format, ready for evaluate.py):
  dense_en             - dense ranking of the English translation
  dense_hyde           - dense ranking of the hypothetical item
  hybrid_multi         - RRF over bm25 + dense PT + dense EN + dense HyDE
  hybrid_multi_rerank  - LLM rerank of hybrid_multi's top-20

Usage:
    python scripts/multiview_search.py
"""

from __future__ import annotations

import csv
import json
import time

import _bootstrap # noqa: F401
import numpy as np

from food_search.config import Config
from food_search.data import load_items, load_queries
from food_search.embedders import QueryCachedEmbedder, embed_corpus_cached, make_embedder
from food_search.rerank import LLMReranker
from food_search.retrieval import BM25Index, SearchResult, rrf_fuse

VIEW_PROMPT = (
    "You expand a Portuguese food-delivery search query into two views. "
    "Reply with JSON only:\n"
    '{"en": "<natural English translation of the query>", '
    '"hyde": "<in Portuguese: name and one-sentence menu description of a '
    'plausible catalog item that perfectly satisfies the query>"}'
)


def get_views(
    client,
    model: str,
    query: str,
) -> dict:
    for attempt in range(4): # backoff on transient failures
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": VIEW_PROMPT},
                    {"role": "user", "content": query},
                ],
            )
            views = json.loads(response.choices[0].message.content)
            if views.get("en") and views.get("hyde"):
                return {"en": str(views["en"]), "hyde": str(views["hyde"])}
            raise ValueError(f"incomplete views: {views}")
        except Exception: # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main() -> None:
    cfg = Config(embedder="openai")
    items = load_items(cfg.items_csv)
    queries = load_queries(cfg.queries_csv)
    embedder = QueryCachedEmbedder(make_embedder("openai", cfg), cfg.artifacts_dir)
    matrix = embed_corpus_cached(embedder, [it.document() for it in items], cfg.artifacts_dir)
    bm25 = BM25Index(items)

    from openai import OpenAI

    client = OpenAI()

    # query views, cached and crash-safe
    views_path = cfg.artifacts_dir / "query_views.json"
    views: dict[str, dict] = json.loads(views_path.read_text(encoding="utf-8")) if views_path.exists() else {}
    for i, query in enumerate(queries, 1):
        if query not in views:
            views[query] = get_views(client, cfg.rerank_model, query)
            views_path.write_text(json.dumps(views, ensure_ascii=False, indent=0), encoding="utf-8")
        if i % 25 == 0:
            print(f"  views {i}/{len(queries)}")
    print(f"query views ready ({len(views)} cached)")

    # rankings per query
    def dense_rank(text: str, kind: str, top_n: int = 50) -> list[int]:
        vec = embedder.embed([text], kind=kind)[0]
        sims = matrix @ vec
        return [int(j) for j in np.argsort(sims)[::-1][:top_n]]

    depth = cfg.candidates_per_channel
    reranker = LLMReranker(cfg.rerank_model)
    writers: dict[str, csv.writer] = {}
    files = {}
    header = ["query", "rank", "itemId", "name", "description", "category", "price", "score", "channels", "image_url"]
    for name in ("dense_en", "dense_hyde", "hybrid_multi", "hybrid_multi_rerank"):
        fh = (cfg.outputs_dir / f"results_{name}_openai.csv").open("w", newline="", encoding="utf-8-sig")
        files[name] = fh
        writers[name] = csv.writer(fh)
        writers[name].writerow(header)

    def emit(name: str, query: str, results: list[SearchResult]) -> None:
        for rank, res in enumerate(results[: cfg.top_k], start=1):
            writers[name].writerow(
                [query, rank, res.item.item_id, res.item.name, res.item.description, res.item.category,
                 res.item.price, f"{res.score:.5f}", "|".join(f"{c}:{r}" for c, r in res.channels.items()),
                 res.item.image_url or ""]
            )

    t0 = time.time()
    for i, query in enumerate(queries, 1):
        view = views[query]
        rankings = {
            "bm25": bm25.rank(query, depth),
            "pt": dense_rank(query, "query"),
            "en": dense_rank(view["en"], "query"),
            # the hypothetical item is document-like, embed it as a passage
            "hyde": dense_rank(view["hyde"], "passage"),
        }
        for name, channels in (("dense_en", ["en"]), ("dense_hyde", ["hyde"])):
            fused = rrf_fuse({c: rankings[c] for c in channels}, k=cfg.rrf_k)
            emit(name, query, [SearchResult(items[j], s, ch) for j, s, ch in fused])

        fused = rrf_fuse(rankings, k=cfg.rrf_k)
        multi = [SearchResult(items[j], s, ch) for j, s, ch in fused]
        emit("hybrid_multi", query, multi)
        emit("hybrid_multi_rerank", query, reranker.rerank(query, multi[: cfg.rerank_pool]))
        if i % 20 == 0:
            print(f"  queries {i}/{len(queries)} ({time.time() - t0:.0f}s)")

    for fh in files.values():
        fh.close()
    print(f"done: 4 systems x {len(queries)} queries in {time.time() - t0:.0f}s -> outputs/")


if __name__ == "__main__":
    main()

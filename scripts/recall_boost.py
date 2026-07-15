"""Recall boosters: index-side doc2query expansion + dense pseudo-relevance
feedback (PRF), on top of the multi-view architecture.

Two ideas from the IR literature, both aimed at retrieval recall (the
measured bottleneck):

* doc2query (Nogueira & Lin) - 21% of items carry no real user search
  terms, which our documents rely on. For those items only, an LLM
  generates 3 short Portuguese queries a customer might type; they are
  appended to the document before indexing. Cached one-time pass.
* dense PRF (Rocchio in embedding space, cf. ANCE-PRF) - retrieve with the
  query vector, mix the centroid of the top-3 document vectors into the
  query (q' = a*q + (1-a)*centroid), retrieve again. Zero API cost.

Systems produced:
  dense_exp            - dense PT query over the expanded corpus (isolates expansion)
  dense_prf            - PRF channel over the original corpus (isolates PRF)
  hybrid_multix        - RRF over bm25 + PT + EN + HyDE + PRF, expanded corpus
  hybrid_multix_rerank - LLM rerank of hybrid_multix's top-20

Reuses the cached query views from multiview_search.py (run that first).

Usage:
    python scripts/recall_boost.py
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

PRF_ALPHA = 0.6 # weight of the original query vector
PRF_DEPTH = 3 # feedback documents

EXPAND_PROMPT = (
    "For each food-delivery catalog item, write 3 short Portuguese search "
    "queries a customer would type to find it (dish type, style, occasion - "
    "not just the literal name). Reply with JSON only: "
    '{"<id>": ["q1", "q2", "q3"], ...} covering every id.'
)


def ensure_expansions(
    client,
    model,
    items,
    path,
) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    pending = [it for it in items if not it.search_terms and it.item_id not in cache]
    print(f"{len(pending)} items need doc2query expansion ({len(cache)} cached)")
    for start in range(0, len(pending), 20):
        batch = pending[start : start + 20]
        payload = [
            {"id": it.item_id, "name": it.name, "category": it.category, "type": it.taxonomy}
            for it in batch
        ]
        for attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": EXPAND_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                returned = json.loads(response.choices[0].message.content)
                break
            except Exception: # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        for it in batch:
            queries = returned.get(it.item_id) or []
            cache[it.item_id] = [str(q).strip().lower() for q in queries if str(q).strip()][:3]
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        if (start // 20) % 10 == 0:
            print(f"  expanded {start + len(batch)}/{len(pending)}")
    return cache


def main() -> None:
    cfg = Config(embedder="openai")
    items = load_items(cfg.items_csv)
    queries = load_queries(cfg.queries_csv)
    views = json.loads((cfg.artifacts_dir / "query_views.json").read_text(encoding="utf-8"))
    embedder = QueryCachedEmbedder(make_embedder("openai", cfg), cfg.artifacts_dir)

    from openai import OpenAI

    client = OpenAI()

    # corpora
    # Original-corpus matrix first (cache hit from earlier runs) - used to
    # isolate the PRF effect. Then mutate items with expansions and index.
    matrix_orig = embed_corpus_cached(embedder, [it.document() for it in items], cfg.artifacts_dir)

    expansions = ensure_expansions(client, cfg.rerank_model, items, cfg.artifacts_dir / "doc_expansions.json")
    for it in items:
        if not it.search_terms and expansions.get(it.item_id):
            it.search_terms = expansions[it.item_id]

    matrix_exp = embed_corpus_cached(embedder, [it.document() for it in items], cfg.artifacts_dir)
    bm25_exp = BM25Index(items)
    print("expanded corpus indexed")

    # systems
    def dense_rank(matrix, vec, top_n=50):
        sims = matrix @ vec
        return [int(j) for j in np.argsort(sims)[::-1][:top_n]]

    def prf_vector(matrix, vec):
        top = dense_rank(matrix, vec, PRF_DEPTH)
        mixed = PRF_ALPHA * vec + (1 - PRF_ALPHA) * matrix[top].mean(axis=0)
        return mixed / np.linalg.norm(mixed)

    reranker = LLMReranker(cfg.rerank_model)
    header = ["query", "rank", "itemId", "name", "description", "category", "price", "score", "channels", "image_url"]
    files, writers = {}, {}
    for name in ("dense_exp", "dense_prf", "hybrid_multix", "hybrid_multix_rerank"):
        fh = (cfg.outputs_dir / f"results_{name}_openai.csv").open("w", newline="", encoding="utf-8-sig")
        files[name], writers[name] = fh, csv.writer(fh)
        writers[name].writerow(header)

    def emit(name, query, results):
        for rank, res in enumerate(results[: cfg.top_k], start=1):
            writers[name].writerow(
                [query, rank, res.item.item_id, res.item.name, res.item.description, res.item.category,
                 res.item.price, f"{res.score:.5f}", "|".join(f"{c}:{r}" for c, r in res.channels.items()),
                 res.item.image_url or ""]
            )

    t0 = time.time()
    for i, query in enumerate(queries, 1):
        view = views[query]
        v_pt = embedder.embed([query], kind="query")[0]
        v_en = embedder.embed([view["en"]], kind="query")[0]
        v_hy = embedder.embed([view["hyde"]], kind="passage")[0]

        rankings = {
            "bm25": bm25_exp.rank(query, 50),
            "pt": dense_rank(matrix_exp, v_pt),
            "en": dense_rank(matrix_exp, v_en),
            "hyde": dense_rank(matrix_exp, v_hy),
            "prf": dense_rank(matrix_exp, prf_vector(matrix_exp, v_pt)),
        }
        emit("dense_exp", query, [SearchResult(items[j], s, ch) for j, s, ch in rrf_fuse({"pt": rankings["pt"]}, k=cfg.rrf_k)])
        prf_orig = dense_rank(matrix_orig, prf_vector(matrix_orig, v_pt))
        emit("dense_prf", query, [SearchResult(items[j], s, ch) for j, s, ch in rrf_fuse({"prf": prf_orig}, k=cfg.rrf_k)])

        fused = rrf_fuse(rankings, k=cfg.rrf_k)
        multi = [SearchResult(items[j], s, ch) for j, s, ch in fused]
        emit("hybrid_multix", query, multi)
        emit("hybrid_multix_rerank", query, reranker.rerank(query, multi[: cfg.rerank_pool]))
        if i % 20 == 0:
            print(f"  queries {i}/{len(queries)} ({time.time() - t0:.0f}s)")

    for fh in files.values():
        fh.close()
    print(f"done: 4 systems x {len(queries)} queries in {time.time() - t0:.0f}s -> outputs/")


if __name__ == "__main__":
    main()

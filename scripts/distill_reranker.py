"""Distill the LLM judge into a small local cross-encoder (Rank-DistiLLM idea).

We already paid for ~3k graded (query, item) pairs while evaluating systems.
This script reuses them as training data:

  1. Split the 100 queries 60/40 into train/test (by QUERY, so the test
     evaluation is on queries the model never saw - no leakage).
  2. Fine-tune cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 on the train pairs
     (explicit torch loop, BCE on soft labels grade/2 in {0, 0.5, 1}).
  3. Rerank the hybrid top-20 for the TEST queries with the tuned model and
     write outputs/results_hybrid_ce-distilled_openai.csv (test queries only -
     compare other systems on the same subset).

Usage:
    python scripts/distill_reranker.py [--epochs 2] [--train-frac 0.6]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time

import _bootstrap # noqa: F401
import torch
from torch.utils.data import DataLoader

from food_search.config import Config
from food_search.data import load_items
from food_search.embedders import QueryCachedEmbedder, embed_corpus_cached, make_embedder
from food_search.rerank import CROSS_ENCODERS, CrossEncoderReranker
from food_search.retrieval import BM25Index, DenseIndex, HybridSearcher

SEP = "\x1e"


def passage_text(item) -> str:
    # Mirrors CrossEncoderReranker._passage so training and inference match.
    parts = [item.name]
    if item.description:
        parts.append(item.description[:200])
    if item.category:
        parts.append(item.category)
    return ". ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=CROSS_ENCODERS["minilm"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(embedder="openai")
    items = load_items(cfg.items_csv)
    items_by_id = {it.item_id: it for it in items}

    # labels and query split
    judgments = json.loads((cfg.artifacts_dir / "judgments.json").read_text(encoding="utf-8"))
    queries = sorted({key.split(SEP)[0] for key in judgments})
    # Split by QUERY, not by pair: pairs within a query are correlated and
    # a pair-level split would leak query identity into training.
    rng = random.Random(args.seed)
    rng.shuffle(queries)
    n_train = int(len(queries) * args.train_frac)
    train_queries, test_queries = set(queries[:n_train]), set(queries[n_train:])
    split_file = cfg.artifacts_dir / "distill_split.json"
    split_file.write_text(
        json.dumps({"train": sorted(train_queries), "test": sorted(test_queries)}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    examples = []
    for key, grade in judgments.items():
        query, item_id = key.split(SEP)
        if query in train_queries and item_id in items_by_id:
            examples.append((query, passage_text(items_by_id[item_id]), grade / 2.0))
    rng.shuffle(examples)
    print(f"{len(queries)} judged queries -> {len(train_queries)} train / {len(test_queries)} test")
    print(f"{len(examples)} training pairs from train queries")

    # fine-tune
    from sentence_transformers import CrossEncoder

    # Start from the mMARCO-pretrained checkpoint; full fine-tune is fine at 118M.
    ce = CrossEncoder(args.base, num_labels=1)
    hf_model, tokenizer = ce.model, ce.tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model.to(device).train()
    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss() # soft labels 0 / 0.5 / 1 = grade / 2

    def collate(batch):
        q, p, y = zip(*batch)
        enc = tokenizer(list(q), list(p), padding=True, truncation=True, max_length=256, return_tensors="pt")
        return enc, torch.tensor(y, dtype=torch.float32)

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    for epoch in range(args.epochs):
        t0, total = time.time(), 0.0
        for step, (enc, labels) in enumerate(loader, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = hf_model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
            if step % 25 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(loader)} loss {total / step:.4f}")
        print(f"epoch {epoch + 1}: mean loss {total / len(loader):.4f} ({time.time() - t0:.0f}s)")

    out_dir = cfg.artifacts_dir / "ce-distilled"
    hf_model.eval()
    ce.save(str(out_dir))
    print(f"Saved tuned cross-encoder -> {out_dir}")

    # rerank the hybrid top-20 for the held-out queries
    embedder = QueryCachedEmbedder(make_embedder("openai", cfg), cfg.artifacts_dir)
    matrix = embed_corpus_cached(embedder, [it.document() for it in items], cfg.artifacts_dir)
    searcher = HybridSearcher(items, BM25Index(items), DenseIndex(matrix, embedder), cfg)
    reranker = CrossEncoderReranker(str(out_dir))

    out_path = cfg.outputs_dir / "results_hybrid_ce-distilled_openai.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["query", "rank", "itemId", "name", "description", "category", "price", "score", "channels", "image_url"]
        )
        for query in sorted(test_queries):
            candidates = searcher.search(query, top_k=cfg.rerank_pool, mode="hybrid")
            for rank, res in enumerate(reranker.rerank(query, candidates)[: cfg.top_k], start=1):
                writer.writerow(
                    [query, rank, res.item.item_id, res.item.name, res.item.description, res.item.category,
                     res.item.price, f"{res.score:.5f}", "|".join(f"{c}:{r}" for c, r in res.channels.items()),
                     res.item.image_url or ""]
                )
    print(f"Test-query results ({len(test_queries)} queries) -> {out_path.name}")
    print("compare systems on the test split only (artifacts/distill_split.json)")


if __name__ == "__main__":
    main()

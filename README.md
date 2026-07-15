# Semantic Food Search

Semantic search that matches natural-language Portuguese food queries
(e.g. *"Batatas fritas de rua carregadas"*) to the most relevant items in a
5,000-item food-delivery catalog, evaluated rigorously without any ground
truth.

**Best system: multi-view hybrid retrieval + LLM reranking.**
nDCG@10 = **0.782**, MRR@10 = **0.94** (a relevant item is ranked #1 for 94%
of queries), P@5 = **0.78**, on a catalog where 28% of items are not food.
Total cost of the entire study: ~$5 of the $75 budget.

```
  query
    |
    |  one cached LLM call expands it into two extra views:
    |  an English translation and a HyDE hypothetical catalog item
    v
  four retrieval channels, run in parallel
    |
    |    BM25          lexical, accent-normalized tokens
    |    dense (PT)    the Portuguese query itself
    |    dense (EN)    the English translation
    |    dense (HyDE)  the hypothetical item, embedded as a passage
    v
  RRF fusion (k = 60)
    |
    v
  top-20  ->  listwise LLM rerank (one call)  ->  top-10
```

Everything below explains this system: what each component does, why it is
there, and the measured gain it earns.

---

## Setup

```bash
pip install -r requirements.txt          # pandas, numpy, rank-bm25, openai, sentence-transformers
echo "OPENAI_API_KEY=sk-..." > .env      # your key; the file is gitignored
# place the two confidential CSVs (never committed) in data/:
#   data/5k_items_curated.csv
#   data/queries.csv
```

Python 3.10+. Without a key, the lexical and local-embedding paths still run
offline via `--embedder local`.

## Reproduce the best system and its evaluation

```bash
# 1. base systems (BM25 / dense / hybrid / hybrid+rerank)
python scripts/run_search.py --embedder openai --systems bm25 dense hybrid hybrid_rerank

# 2. the winning system: EN + HyDE views, 4-channel fusion, listwise rerank
python scripts/multiview_search.py

# 3. evaluate everything on one shared, judged pool
python scripts/evaluate.py outputs/results_*.csv --audit-sample

# interactive demo (shows per-channel provenance for each hit)
python scripts/demo.py --embedder openai
python -m pytest tests/ -q
```

---

## 1. The problem

Return a ranked top-10 per query and prove the ranking is good. Three facts
from the data drive the whole design:

1. **Queries are conceptual, not lexical** - "almoco estilo havaiano",
   "estilo caminhao de tacos" describe intent, not item names. Keyword
   matching alone cannot solve this.
2. **The catalog is adversarial** - ~28% of items are not food (beauty,
   household, pet, electronics) and lexical overlap misleads (a seasoning
   named "Nordeste" traps the query "comida de rua do nordeste").
3. **There is no ground truth** - so measuring relevance is half the task,
   and every design choice below is backed by a number from Section 7.

Each item carries `itemMetadata` (name, category, description, price,
taxonomy, dietary flags, images) and `itemProfile` (behavioral data,
including the real search terms users typed before buying the item).

**Output.** For each of the 100 queries the system writes a ranked
**top-10** to `outputs/results_<system>_openai.csv` (the assignment asks for
at least top-5), with item id, name, score and per-channel provenance.

**Assumptions** (stated, per the brief):
- Queries and catalog are both Portuguese; the English translation is a
  retrieval *view* (Section 3), not a preprocessing step - the embeddings
  are multilingual either way.
- The 28% non-food items are kept in the index as realistic distractors;
  the reranker and judge treat non-food as never relevant.
- 3 duplicate `itemId`s are dropped (first occurrence kept).
- Relevance is judged on text metadata only; images are surfaced
  (`image_url`) but not used for matching.

---

## 2. The document: what we index per item

One flat text document per item, highest-signal field first:

```
name . description . Categoria: ... . Tipo: <taxonomy> . <dietary words> . Buscado como: <user search terms>
```

The distinctive ingredient is **`Buscado como:` - the real search terms
users typed before buying that item**, taken from `itemProfile`. This is
free, human-generated query-language expansion (the doc2query idea,
[Nogueira et al., 2019](https://arxiv.org/abs/1904.08375)) that bridges
catalog vocabulary ("Sanduiche Bauru") and query vocabulary ("lanche estilo
Bauru"). Terms are noisy, so they are capped at the 8 most frequent and
placed last, where they cannot outweigh the name. Tokenization for BM25
strips accents ("Pao" matches "pão") with a deliberately tiny stopword list
so short dish names survive.

---

## 3. Retrieval: four complementary views of the query

The core idea of the best system: **one query, four retrieval channels**,
each covering a different failure mode.

**Channel 1 - BM25 (lexical).**
[Robertson and Zaragoza, 2009](https://dl.acm.org/doi/10.1561/1500000019).
Catches exact dish names, brands and rare tokens embeddings underrepresent.
Alone: nDCG@10 = 0.332 (weak, confirming the problem is semantic).

**Channel 2 - dense on the Portuguese query.**
Bi-encoder retrieval ([Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906))
with OpenAI `text-embedding-3-small`; exact cosine over the 5k-vector matrix
(a NumPy matmul - an ANN index would add complexity for no benefit at this
scale). Catches paraphrase and concept. Alone: 0.539.

**Channel 3 - dense on the English translation of the query.**
Multilingual embedding models are trained on English-heavy data, so the
English subspace is better organized ([Wang et al., 2024](https://arxiv.org/abs/2402.05672)).
We embed an English translation of the query against the same Portuguese
documents (cross-lingual matching is exactly what these models are built
for). Alone: 0.550 - it beats the native Portuguese query (+0.011 nDCG,
+0.036 MRR).

**Channel 4 - dense on a HyDE hypothetical item.**
HyDE ([Gao et al., 2022](https://arxiv.org/abs/2212.10496)): instead of
matching a short query to documents (hard geometry), have the LLM write a
plausible catalog item that would satisfy the query and match that
*document to documents* (well-conditioned geometry). We embed the
hypothetical item as a *passage*, placing it on the document side of the
space. Alone: 0.574 - the strongest single view, and +38% relative on
strict P@5, because hypothetical items pull back *exact* matches, not just
related ones.

Channels 3 and 4 are produced by **one cached gpt-4.1-mini call per query**
(the call returns both the translation and the hypothetical item), so the
two extra views cost a fraction of a cent and are computed once.

---

## 4. Fusion: Reciprocal Rank Fusion

The four channels are combined with RRF
([Cormack et al., 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)):

```
score(d) = sum over channels of 1 / (60 + rank_channel(d))
```

RRF works on ranks, not scores, so BM25 and cosine similarities never need
calibration and there is nothing to tune - the right choice when no labels
exist. Each result keeps its per-channel rank provenance (the demo prints
`bm25#5, hyde#1`). Fused, the four-channel system scores **0.600 nDCG@10 -
higher than any individual channel**, proving the views are complementary.

---

## 5. Reranking: one listwise LLM pass

Retrieval scores cannot arbitrate fine intent (packaged chips vs a plate of
loaded fries both match "batatas fritas"). RankGPT
([Sun et al., 2023](https://arxiv.org/abs/2304.09542)) shows an LLM that
sees all candidates jointly outperforms independent scoring. One
gpt-4.1-mini call takes the fused top-20 as compact JSON cards and returns
them in relevance order; invalid/duplicate ids are dropped, omitted items
are re-appended in fused order, and any failure falls back to the fused
ranking (reranking can only help). Applied to the four-view pool this is the
**best system: 0.782 nDCG@10, MRR 0.94, P@5 0.78.**

Why it works so well here: an oracle that reranks the two-channel pool
perfectly tops out at ~0.82, i.e. reranking headroom was nearly exhausted -
the remaining wins had to come from *recall*, which is exactly what the
extra views provide. Feeding the same reranker the four-view pool instead
of the two-channel pool lifts nDCG from 0.698 to **0.782 (+0.084)**.

---

## 6. Results

All systems judged once on a single shared pool (Section 7); nDCG is
pool-relative, so numbers are comparable within this table.

| System | nDCG@10 | MRR@10 | P@5 | P@5 strict |
|---|---|---|---|---|
| BM25 only | 0.332 | 0.575 | 0.328 | 0.098 |
| Dense (local e5-small, free/offline) | 0.416 | 0.644 | 0.452 | 0.128 |
| Hybrid (BM25 + dense PT, 2-channel) | 0.503 | 0.782 | 0.540 | 0.176 |
| Dense (PT query) | 0.539 | 0.760 | 0.576 | 0.178 |
| Dense (EN translation view) | 0.550 | 0.796 | 0.574 | 0.180 |
| Dense (HyDE view) | 0.574 | 0.763 | 0.578 | 0.246 |
| Hybrid multi-view (4-channel) | 0.600 | 0.843 | 0.630 | 0.216 |
| Hybrid + LLM rerank (2-channel) | 0.698 | 0.900 | 0.744 | 0.252 |
| **Hybrid multi-view + LLM rerank (best)** | **0.782** | **0.940** | **0.778** | **0.314** |

The story in one line: **semantics beat lexical (+0.21), multi-view beats
single-view (+0.06), reranking converts the richer pool into a strong final
ranking (+0.08).**

---

## 7. How we know it is good: evaluation without ground truth

A scaled-down TREC pooling protocol - the standard IR answer to "no labels":

1. **Pool.** Every system contributes its top-10 per query; their union is
   the judgment pool (~4.2k pairs instead of 500k).
2. **Judge.** gpt-4.1-mini grades each pooled (query, item) pair once on a
   0/1/2 rubric (2 = satisfies intent, 1 = partial, 0 = irrelevant; non-food
   is never relevant), temperature 0. LLM judges track human searcher
   preference closely ([Thomas et al., 2024](https://arxiv.org/abs/2309.10621)).
   Grades are cached, so all systems are compared on identical judgments and
   every new experiment costs cents.
3. **Metrics.** nDCG@10 ([Jarvelin and Kekalainen, 2002](https://dl.acm.org/doi/10.1145/582415.582418))
   against the ideal ranking of the pool, MRR@10, P@5 lenient and strict.
4. **Validate the judge.** `--audit-sample` exports a stratified, shuffled
   sample for a human to grade; `--audit-score` reports human/judge
   agreement. On a 45-pair sample the human reviewer agreed with the judge
   on **84%** of relevant-vs-not decisions (71% on the exact 0/1/2 grade),
   with disagreements concentrated in boundary 0-vs-1 cases and the judge
   being marginally stricter - i.e. the judge is a trustworthy, slightly
   conservative relevance signal. This closes the loop on "is the judge
   trustworthy?".

Stated biases and mitigations: judge and reranker share a model family (all
systems share identical judgments, so comparisons stay fair; the human audit
bounds absolute error); pooling is blind to never-retrieved items (mitigated
by pooling many diverse systems); nDCG is pool-relative (comparable within a
table, not across pools).

---

## 8. What else was tried (and honestly did not win)

Rigor means reporting the dead ends; the harness caught each one:

- **Local reranker distillation (cross-encoder, LoRA, DPO).** Fine-tuning a
  118M/568M/1.5B model on the judge labels to replace the API reranker.
  Distillation rescued a *weak* cross-encoder (0.475 -> 0.595) but no local
  method reached the API reranker; with only 60 training queries, the
  bottleneck is label quantity, not model size. (Scripts in `scripts/` and
  `colab/`.)
- **doc2query + pseudo-relevance feedback.** Expanding thin-document items
  and adding a PRF channel to boost recall. Both slightly *hurt* (multi-view
  0.600 -> 0.587), because the thin items are disproportionately the 28%
  non-food distractors - making them more retrievable adds noise. A clean
  lesson that a technique's value depends on corpus structure, not pedigree.

These are documented so the winning system's choices are defensible by
contrast.

---

## 9. Cost

| Step | Model | Cost |
|---|---|---|
| Embed the 5k catalog (cached) | text-embedding-3-small | < $0.05 |
| Query embeddings + views (EN + HyDE, cached) | 3-small / gpt-4.1-mini | ~ $0.50 |
| Listwise reranking, 100 queries | gpt-4.1-mini | ~ $0.50 |
| Judge the full pool (cached) | gpt-4.1-mini | ~ $3.00 |
| **Total** | | **~ $5 of $75** |

---

## 10. Limitations and future work

- **Images unused** - `image_url` is surfaced but not matched on; a
  CLIP-style visual channel is the clearest untapped signal.
- **RRF untuned** - the accumulated judge labels now make learned fusion
  weights measurable.
- **Judge validated by one human sample**, not a panel; in production the
  `itemProfile` conversion/reorder signals are the real ground truth.
- **100 queries is small** - read differences under ~0.02 nDCG as noise; the
  reported gaps are several times that.

---

## Repository layout

```
src/food_search/
  config.py        # paths, models, constants; .env loading
  data.py          # CSV/JSON parsing, document builder, PT tokenization
  embedders.py     # OpenAI + local backends, corpus and query caches
  retrieval.py     # BM25Index, DenseIndex, RRF fusion, HybridSearcher
  rerank.py        # listwise LLM reranker (+ local cross-encoder)
  evaluation.py    # judgment store, LLM judge, nDCG/MRR/P@k, audit sample
scripts/
  run_search.py        # base systems (bm25 / dense / hybrid / +rerank)
  multiview_search.py  # the best system: EN + HyDE views, fusion, rerank
  evaluate.py          # pooled judging, metrics table, judge audit
  demo.py              # interactive CLI with per-channel provenance
  distill_reranker.py, recall_boost.py, export_colab_data.py  # ablations (Section 8)
tests/                 # unit tests for fusion, metrics, normalization
```

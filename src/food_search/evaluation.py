"""Evaluation without ground truth: pooled LLM-as-judge + graded IR metrics.

Method (a scaled-down TREC pooling protocol):
  1. Every system variant contributes its top-k per query to a shared pool.
  2. An LLM judge grades each (query, pooled item) pair once on a 0-2 scale.
     Grades are cached, so systems are compared on identical judgments and
     re-evaluation is free.
  3. Standard graded metrics (nDCG@10, MRR@10, P@5) are computed per system
     from the shared grade table.
  4. A stratified audit sample is exported for human validation of the judge.

Grades: 2 = clearly satisfies the query intent; 1 = partially relevant
(related dish or category, missing a key attribute); 0 = not relevant.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .data import Item

JUDGE_SYSTEM_PROMPT = (
    "You are a strict relevance judge for a Brazilian food-delivery search engine. "
    "For a Portuguese search query, grade each candidate item:\n"
    "2 = clearly satisfies the query intent (dish type, style, cuisine and key attributes match)\n"
    "1 = partially relevant (related dish or category, but a key attribute is missing or unclear)\n"
    "0 = not relevant (different dish, or not food)\n"
    "Judge only what the item data says; do not guess missing attributes in the item's favor. "
    'Respond with JSON only: {"grades": {"<id>": <0|1|2>, ...}} covering every id.'
)


class JudgmentStore:
    """Grade cache keyed by (query, item_id) in a JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self._grades: dict[str, int] = {}
        if path.exists():
            self._grades = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _key(query: str, item_id: str) -> str:
        # \x1e (ASCII record separator) cannot appear in queries or ids.
        return f"{query}\x1e{item_id}"

    def get(self, query: str, item_id: str) -> int | None:
        return self._grades.get(self._key(query, item_id))

    def put(self, query: str, item_id: str, grade: int) -> None:
        self._grades[self._key(query, item_id)] = grade

    def save(self) -> None:
        # Called after every judged query: progress survives crashes/retries.
        self.path.write_text(
            json.dumps(self._grades, ensure_ascii=False, indent=0), encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(self._grades)


class LLMJudge:
    def __init__(
        self,
        store: JudgmentStore,
        model: str = "gpt-4.1-mini",
    ):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.store = store

    def grade_pool(self, query: str, items: list[Item]) -> dict[str, int]:
        """Grade all pooled items for a query in one call; cached pairs are skipped."""
        grades: dict[str, int] = {}
        pending: list[Item] = []
        for item in items:
            cached = self.store.get(query, item.item_id)
            if cached is None:
                pending.append(item)
            else:
                grades[item.item_id] = cached

        if pending:
            payload = [
                {
                    "id": it.item_id,
                    "name": it.name,
                    "description": it.description[:200],
                    "category": it.category,
                    "type": it.taxonomy,
                }
                for it in pending
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nItems:\n{json.dumps(payload, ensure_ascii=False)}",
                    },
                ],
            )
            returned = json.loads(response.choices[0].message.content).get("grades", {})
            for it in pending:
                # missing/malformed grades count as 0 (conservative)
                grade = returned.get(it.item_id)
                grade = int(grade) if grade in (0, 1, 2, "0", "1", "2") else 0
                grades[it.item_id] = grade
                self.store.put(query, it.item_id, grade)
            self.store.save()
        return grades


# Graded IR metrics: `grades` is the list of grades in result order.

def dcg(grades: list[int], k: int) -> float:
    # Exponential gain (2^g - 1) weights a grade-2 hit 3x a grade-1 hit.
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(
    grades: list[int],
    pool_grades: list[int],
    k: int = 10,
) -> float:
    """nDCG against the ideal ranking of the *pooled* judgments for the query."""
    ideal = dcg(sorted(pool_grades, reverse=True), k)
    return dcg(grades, k) / ideal if ideal > 0 else 0.0


def mrr_at_k(
    grades: list[int],
    k: int = 10,
    min_grade: int = 1,
) -> float:
    for i, g in enumerate(grades[:k]):
        if g >= min_grade:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    grades: list[int],
    k: int = 5,
    min_grade: int = 1,
) -> float:
    top = grades[:k]
    return sum(g >= min_grade for g in top) / k if top else 0.0


def summarize_system(
    per_query_grades: dict[str, list[int]],
    pool: dict[str, list[int]],
) -> dict[str, float]:
    """Average metrics over queries. `pool[q]` holds all pooled grades for q."""
    n = len(per_query_grades)
    agg = {"nDCG@10": 0.0, "MRR@10": 0.0, "P@5": 0.0, "P@5_strict": 0.0}
    for query, grades in per_query_grades.items():
        agg["nDCG@10"] += ndcg_at_k(grades, pool[query], 10)
        agg["MRR@10"] += mrr_at_k(grades, 10, min_grade=1)
        agg["P@5"] += precision_at_k(grades, 5, min_grade=1)
        agg["P@5_strict"] += precision_at_k(grades, 5, min_grade=2)
    return {metric: round(value / n, 4) for metric, value in agg.items()}


def export_audit_sample(
    store: JudgmentStore, items_by_id: dict[str, Item], out_csv: Path, n: int = 45, seed: int = 13
) -> None:
    """Stratified sample (equal counts per grade) for human validation of the judge."""
    import csv

    by_grade: dict[int, list[tuple[str, str]]] = {0: [], 1: [], 2: []}
    for key, grade in store._grades.items():
        query, item_id = key.split("\x1e")
        by_grade[grade].append((query, item_id))
    rng = random.Random(seed)
    rows = []
    for grade, pairs in by_grade.items():
        for query, item_id in rng.sample(pairs, min(n // 3, len(pairs))):
            item = items_by_id.get(item_id)
            rows.append(
                {
                    "query": query,
                    "item_name": item.name if item else "?",
                    "item_description": item.description if item else "?",
                    "item_category": item.category if item else "?",
                    "judge_grade": grade,
                    "human_grade": "",
                }
            )
    # Shuffle so the annotator cannot infer the judge's grade from row order.
    rng.shuffle(rows)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

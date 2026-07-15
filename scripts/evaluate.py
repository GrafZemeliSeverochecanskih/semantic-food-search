"""Evaluate result files with the pooled LLM-judge protocol.

Usage:
    python scripts/evaluate.py outputs/results_*.csv
    python scripts/evaluate.py outputs/results_hybrid_openai.csv --audit-sample

Every results CSV contributes its items to a shared per-query pool; the judge
grades each pooled (query, item) pair once (cached in artifacts/judgments.json);
metrics are computed per system on those shared grades.

--audit-sample exports a stratified sample to outputs/judge_audit_sample.csv.
Fill in the human_grade column and re-run with --audit-score to report
human/judge agreement.
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import _bootstrap # noqa: F401

from food_search.config import Config
from food_search.data import load_items
from food_search.evaluation import (
    JudgmentStore,
    LLMJudge,
    export_audit_sample,
    summarize_system,
)


def read_results(path: Path) -> dict[str, list[str]]:
    """query -> [itemId, ...] in rank order."""
    runs: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            runs[row["query"]].append(row["itemId"])
    return runs


def audit_score(path: Path) -> None:
    with path.open(encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["human_grade"].strip() != ""]
    if not rows:
        print("No human grades filled in yet.")
        return
    # Exact = same 0/1/2 grade; binary = agree on relevant (>=1) vs not.
    exact = sum(int(r["human_grade"]) == int(r["judge_grade"]) for r in rows)
    binary = sum((int(r["human_grade"]) >= 1) == (int(r["judge_grade"]) >= 1) for r in rows)
    print(f"Judge-human agreement on {len(rows)} pairs: exact {exact/len(rows):.0%}, relevant/not {binary/len(rows):.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", help="results CSVs (globs ok)")
    parser.add_argument("--audit-sample", action="store_true")
    parser.add_argument("--audit-score", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    audit_csv = cfg.outputs_dir / "judge_audit_sample.csv"
    if args.audit_score:
        audit_score(audit_csv)
        return

    paths = sorted({Path(p) for pattern in args.results for p in glob.glob(pattern)})
    if not paths:
        parser.error("no results files matched")

    items = load_items(cfg.items_csv)
    items_by_id = {it.item_id: it for it in items}
    systems = {p.stem.removeprefix("results_"): read_results(p) for p in paths}

    # pool: union of every system's results per query
    pool: dict[str, list[str]] = defaultdict(list)
    for runs in systems.values():
        for query, ids in runs.items():
            for item_id in ids:
                if item_id not in pool[query]:
                    pool[query].append(item_id)

    # judge the pool, reusing cached grades
    store = JudgmentStore(cfg.artifacts_dir / "judgments.json")
    judge = LLMJudge(store, cfg.judge_model)
    total_pairs = sum(len(ids) for ids in pool.values())
    print(f"Pool: {len(pool)} queries, {total_pairs} (query,item) pairs; {len(store)} already judged")
    for i, (query, ids) in enumerate(pool.items(), 1):
        judge.grade_pool(query, [items_by_id[i_] for i_ in ids if i_ in items_by_id])
        if i % 20 == 0:
            print(f"  judged {i}/{len(pool)} queries")

    # metrics per system on the shared grades ("or 0" covers any judge miss)
    pool_grades = {
        q: [store.get(q, item_id) or 0 for item_id in ids] for q, ids in pool.items()
    }
    report_lines = [
        "| system | nDCG@10 | MRR@10 | P@5 | P@5 (strict) |",
        "|---|---|---|---|---|",
    ]
    for name, runs in sorted(systems.items()):
        per_query = {q: [store.get(q, i_) or 0 for i_ in ids] for q, ids in runs.items()}
        m = summarize_system(per_query, pool_grades)
        report_lines.append(
            f"| {name} | {m['nDCG@10']:.3f} | {m['MRR@10']:.3f} | {m['P@5']:.3f} | {m['P@5_strict']:.3f} |"
        )
        print(f"{name:28s} nDCG@10={m['nDCG@10']:.3f} MRR@10={m['MRR@10']:.3f} P@5={m['P@5']:.3f} P@5_strict={m['P@5_strict']:.3f}")

    report = cfg.outputs_dir / "eval_report.md"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report -> {report}")

    if args.audit_sample:
        export_audit_sample(store, items_by_id, audit_csv)
        print(f"Audit sample -> {audit_csv} (fill human_grade, then --audit-score)")


if __name__ == "__main__":
    main()

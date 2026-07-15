"""Unit tests for the pure logic: normalization, RRF fusion, graded metrics.

These are the functions where a silent bug would corrupt every reported
number, so they are the ones worth pinning down.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_search.data import normalize, tokenize
from food_search.evaluation import mrr_at_k, ndcg_at_k, precision_at_k
from food_search.retrieval import rrf_fuse


def test_normalize_strips_accents():
    assert normalize("Pão de Queijo à moda") == "pao de queijo a moda"


def test_tokenize_removes_stopwords():
    assert tokenize("Pizza de massa fina") == ["pizza", "massa", "fina"]


def test_rrf_single_channel_preserves_order():
    fused = rrf_fuse({"bm25": [7, 3, 9]})
    assert [idx for idx, _, _ in fused] == [7, 3, 9]


def test_rrf_rewards_agreement():
    # doc 5 is mid-ranked in both channels; docs 1 and 2 top one channel each.
    fused = rrf_fuse({"bm25": [1, 5, 8], "dense": [2, 5, 9]}, k=1)
    assert fused[0][0] == 5
    assert fused[0][2] == {"bm25": 2, "dense": 2}


def test_ndcg_perfect_ranking_is_one():
    # Best-first order must equal the ideal DCG of the same pool exactly.
    assert ndcg_at_k([2, 1, 0], pool_grades=[0, 1, 2], k=10) == 1.0


def test_ndcg_worst_ranking_below_one():
    assert ndcg_at_k([0, 1, 2], pool_grades=[0, 1, 2], k=10) < 1.0


def test_mrr_first_relevant_position():
    assert mrr_at_k([0, 0, 1]) == 1 / 3
    assert mrr_at_k([0, 0, 0]) == 0.0


def test_precision_thresholds():
    grades = [2, 1, 0, 0, 0]
    assert math.isclose(precision_at_k(grades, 5, min_grade=1), 0.4)
    assert math.isclose(precision_at_k(grades, 5, min_grade=2), 0.2)

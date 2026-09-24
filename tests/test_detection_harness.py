"""Tests for the frozen evaluation harness and split grouping."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fractalsig.detection import metrics as M
from fractalsig.detection.splits import group_key


@pytest.fixture(scope="module")
def scores():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(1500)].astype(int)
    s = np.r_[rng.normal(0, 1, 500), rng.normal(1.5, 1, 1500)]
    return y, s


def test_auroc_matches_sklearn(scores):
    y, s = scores
    assert M.auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_tpr_at_fpr_bounds_and_monotonicity(scores):
    y, s = scores
    t1, t5 = M.tpr_at_1pct_fpr(y, s), M.tpr_at_5pct_fpr(y, s)
    assert 0.0 <= t1 <= t5 <= 1.0


def test_perfect_separation():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.2, 0.8, 0.9])
    assert M.auroc(y, s) == 1.0 and M.tpr_at_1pct_fpr(y, s) == 1.0


def test_delong_identical_scores_gives_no_difference(scores):
    y, s = scores
    auc_a, auc_b, z, p = M.delong_test(y, s, s)
    assert auc_a == pytest.approx(auc_b)
    assert np.isnan(z) or abs(z) < 1e-9
    assert np.isnan(p) or p > 0.99


def test_bootstrap_ci_contains_point_and_is_deterministic(scores):
    y, s = scores
    a = M.bootstrap_ci(M.auroc, y, s, n_boot=200)
    b = M.bootstrap_ci(M.auroc, y, s, n_boot=200)
    point, lo, hi, n_valid = a
    assert a == b and lo <= point <= hi and n_valid == 200


def test_group_key_links_human_and_machine_rows_of_one_article():
    human = {"id": "big_patent_00000-of-00256__0_human", "domain": "big_patent", "Label": "Human"}
    llm = {"id": "big-p-gemma-2b-extend-0.0-00000-of-00256--0", "domain": "big_patent", "Label": "gemma_2b"}
    assert group_key(human) == group_key(llm) == ("big_patent", "00000-of-00256__0")


def test_group_key_rejects_unparseable_id():
    assert group_key({"id": "garbage", "domain": "wikipedia", "Label": "Human"}) is None

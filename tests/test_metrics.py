import pytest

from trajlens.metrics import cohens_kappa, compare, confusion, kappa_ci


def test_kappa_textbook_example():
    # 20 agree-yes, 5 a-yes/b-no, 10 a-no/b-yes, 15 agree-no -> kappa 0.4
    a = ["PASSED"] * 25 + ["FAILED"] * 25
    b = ["PASSED"] * 20 + ["FAILED"] * 5 + ["PASSED"] * 10 + ["FAILED"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_kappa_perfect_agreement():
    a = ["PASSED", "FAILED"] * 5
    assert cohens_kappa(a, a) == 1.0
    assert cohens_kappa(["PASSED"] * 4, ["PASSED"] * 4) == 1.0


def test_confusion_treats_failed_as_positive():
    c = confusion(["FAILED", "FAILED", "PASSED", "PASSED"], ["FAILED", "PASSED", "FAILED", "PASSED"])
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert c["precision"] == c["recall"] == 0.5


def test_ci_contains_point_estimate_and_is_reproducible():
    a = ["PASSED", "FAILED", "FAILED", "PASSED", "FAILED", "PASSED", "FAILED", "FAILED"]
    b = ["PASSED", "PASSED", "FAILED", "PASSED", "FAILED", "FAILED", "FAILED", "PASSED"]
    lo, hi = kappa_ci(a, b)
    assert lo <= cohens_kappa(a, b) <= hi
    assert kappa_ci(a, b) == (lo, hi)


def test_compare_reports_agreement():
    r = compare(["FAILED", "PASSED"], ["FAILED", "PASSED"])
    assert r["agree"] == 2 and r["kappa"] == 1.0

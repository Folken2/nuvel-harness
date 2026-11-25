"""The overage rule on its own."""

from decimal import Decimal as D

from harness.ledger.overage import overage


def test_under_the_limit_there_is_none():
    assert overage(D("10"), D("5"), D("25")) == 0
    assert overage(D("20"), D("5"), D("25")) == 0


def test_the_crossing_call_counts_only_its_part_above():
    assert overage(D("24.90"), D("0.08"), D("25")) == 0
    assert overage(D("24.98"), D("0.08"), D("25")) == D("0.06")


def test_once_at_the_limit_all_of_a_call_is_over():
    assert overage(D("25"), D("0.08"), D("25")) == D("0.08")
    assert overage(D("31.40"), D("1"), D("25")) == D("1")


def test_no_limit_means_no_overage():
    assert overage(D("1000"), D("5"), None) == 0

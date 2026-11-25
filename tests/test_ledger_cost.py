"""The ledger's cost rules, against the acceptance tests of usage-and-analytics.md.

Each test names the AT it covers. Amounts are Decimal and compared exactly.
"""

from decimal import Decimal as D

import pytest

from harness.billing.evidence import provider_id, sanitise_provider_usage
from harness.billing.pricing import (
    CUT_OFF,
    NO_ESTIMATE,
    NOT_CONTACTED,
    REFUSED,
    FallbackRates,
    Rates,
    Units,
    estimate_tokens,
    price_aborted,
    price_call,
)
from harness.billing.rollup import (
    CostSplit,
    Settled,
    interaction_source,
    is_unresolved,
    quality,
    split,
    totals,
)
from harness.billing.tokens import TokenUsage, normalise, summed

TOLERANCE = D("0.000000000001")


def rates(**per_1m):
    """Rates in USD per million tokens, by snapshot name (``cacheRead=0.3``)."""
    return Rates.from_mapping({f"{k}Per1M": v for k, v in per_1m.items()})


def tokens(input_tokens=0, output_tokens=0, **kw):
    return normalise(input_tokens=input_tokens, output_tokens=output_tokens, **kw)


# --- AT-17 (UG-17) token normalisation ---------------------------------------


def test_at17_input_reported_without_cache_gets_cache_added():
    usage = normalise(input_tokens=1000, input_includes_cache=False, cache_read=500, cache_write=100)
    assert (usage.input_total, usage.ordinary_input) == (1600, 1000)
    assert (usage.cache_read, usage.cache_write) == (500, 100)


def test_at17_input_reported_with_cache_gives_the_same_buckets():
    assert normalise(
        input_tokens=1600, input_includes_cache=True, cache_read=500, cache_write=100
    ) == normalise(input_tokens=1000, input_includes_cache=False, cache_read=500, cache_write=100)


def test_at17_thoughts_reported_apart_are_added_once():
    usage = normalise(output_tokens=100, reasoning=40, output_includes_reasoning=False)
    assert (usage.output_total, usage.reasoning) == (140, 40)


def test_at17_reasoning_inside_output_is_not_added_again():
    usage = normalise(output_tokens=140, reasoning=40)
    assert (usage.output_total, usage.reasoning) == (140, 40)


def test_at17_last_step_only_is_flagged():
    usage = normalise(input_tokens=10, output_tokens=5, incomplete=True)
    priced = price_call(usage=usage, rates=rates(input=3, output=15))
    assert priced.pricing_missing


def test_ug17_steps_sum_and_caps_hold():
    steps = [tokens(100, 10), tokens(200, 20, cache_read=50)]
    usage = summed(steps)
    assert (usage.input_total, usage.output_total, usage.cache_read) == (300, 30, 50)
    assert summed([]) is None
    # Cache read is capped at input total, write at what remains.
    capped = normalise(input_tokens=100, cache_read=80, cache_write=50, cache_write_1h=90)
    assert (capped.cache_read, capped.cache_write, capped.cache_write_1h) == (80, 20, 20)
    assert capped.ordinary_input == 0
    # Total is the reported one when above 0.
    assert normalise(input_tokens=10, output_tokens=5, total=99).total == 99
    assert normalise(input_tokens=10, output_tokens=5).total == 15
    # Garbage counts are 0, never negative.
    assert normalise(input_tokens=-5, output_tokens=float("nan")).input_total == 0


# --- AT-18 (UG-18) which cost applies -----------------------------------------


def test_at18_reported_cost_wins_over_rates():
    priced = price_call(usage=tokens(10_000, 1_000), rates=rates(input=3, output=20), reported_cost=0.01)
    assert (priced.cost_usd, priced.source) == (D("0.01"), "provider_reported")


def test_at18_each_call_uses_its_own_best_source():
    first = price_call(usage=tokens(1000, 100), rates=rates(input=3, output=15), reported_cost=0.01)
    # 6,000 x 2 + 400 x 20 per million = 0.012 + 0.008 = 0.02
    second = price_call(usage=tokens(6000, 400), rates=rates(input=2, output=20))
    assert second.cost_usd == D("0.02") and second.source == "model_pricing"
    calls = [Settled(first.cost_usd, first.source), Settled(second.cost_usd, second.source)]
    result = totals(calls)
    assert result.cost_usd == D("0.03")
    assert result.source == "mixed"
    assert first.cost_usd == D("0.01")


# --- AT-19 (UG-19) the worked example ---------------------------------------


def test_at19_worked_example_is_exact():
    usage = normalise(
        input_tokens=10_000,
        cache_read=6_000,
        cache_write=1_000,
        cache_write_1h=200,
        output_tokens=2_000,
        reasoning=500,
    )
    model = rates(input=3, cacheRead=0.30, cacheWrite=3.75, cacheWrite1h=6, output=15)
    priced = price_call(usage=usage, rates=model)
    assert priced.breakdown["input"] == D("0.009")
    assert priced.breakdown["cached_input"] == D("0.0018")
    assert priced.breakdown["cache_write"] == D("0.003")
    assert priced.breakdown["cache_write_1h"] == D("0.0012")
    assert priced.breakdown["output"] == D("0.03")
    assert str(priced.cost_usd) == "0.045000000000"
    assert priced.source == "model_pricing"
    assert not priced.pricing_missing
    assert {"input", "cache_read", "cache_write", "cache_write_1h", "output"} <= set(priced.snapshot)


def test_ug19_per_token_rate_accepted_and_per_million_wins():
    per_token = Rates.from_mapping({"inputPerToken": 0.000003, "outputPer1M": 15})
    assert per_token.input == D("3.000000")
    both = Rates.from_mapping({"inputPerToken": 0.000001, "inputPer1M": 3})
    assert both.input == D(3)


def test_ug19_modality_input_at_its_own_rate():
    usage = normalise(input_tokens=1_000, input_image=400, output_tokens=0)
    priced = price_call(usage=usage, rates=rates(input=1, inputImage=10, output=1))
    # 600 text x 1 + 400 image x 10 per million
    assert priced.cost_usd == D("0.0046")
    assert not priced.pricing_missing


# --- AT-20 (UG-20) a missing price is never free -----------------------------


def test_at20_empty_catalogue_uses_15_and_60():
    priced = price_call(usage=tokens(1000, 500), rates=None, fallback=FallbackRates())
    assert priced.cost_usd == D("0.045")
    assert priced.source == "fallback_pricing"
    assert priced.pricing_missing
    assert set(priced.missing) == {"inputTokens", "outputTokens"}


def test_at20_highest_catalogue_rates():
    fallback = FallbackRates.from_catalogue(["1", "5", None, "0"], [25, "10", None])
    assert (fallback.input, fallback.output) == (D(5), D(25))
    priced = price_call(usage=tokens(1000, 500), rates=Rates(), fallback=fallback)
    assert priced.cost_usd == D("0.0175")
    assert FallbackRates.from_catalogue([], []) == FallbackRates()


def test_at20_explicit_zero_is_free():
    priced = price_call(usage=tokens(1000, 500), rates=rates(input=0, output=0))
    assert (priced.cost_usd, priced.source, priced.pricing_missing) == (D(0), "free", False)


def test_at20_missing_cache_rate_falls_back_to_input_and_is_flagged():
    priced = price_call(usage=tokens(1000, 0, cache_read=1000), rates=rates(input=3, output=15))
    assert priced.breakdown["cached_input"] == D("0.003")
    assert priced.missing == ("cacheReadTokens",)
    assert priced.source == "model_pricing"
    assert priced.pricing_missing


def test_mc38_zero_input_with_blank_cache_is_zero_and_not_flagged():
    priced = price_call(usage=tokens(1000, 0, cache_read=1000), rates=rates(input=0, output=0))
    assert priced.cost_usd == D(0) and not priced.pricing_missing


def test_at20_non_usd_rates_are_flagged():
    model = Rates.from_mapping({"inputPer1M": 3, "outputPer1M": 15, "currency": "EUR"})
    priced = price_call(usage=tokens(1000, 500), rates=model)
    assert "tokenCurrency" in priced.missing
    assert priced.source == "fallback_pricing"


def test_at20_only_a_reported_non_standard_tier_is_flagged():
    model = rates(input=3, output=15)
    flex = price_call(usage=tokens(1000, 500), rates=model, service_tier="flex")
    assert "non_standard_service_tier" in flex.missing
    for tier in ("default", "standard", None):
        assert price_call(usage=tokens(1000, 500), rates=model, service_tier=tier).missing == ()


def test_ug20_12h_cache_write_without_report_is_an_estimate():
    usage = normalise(input_tokens=1000, cache_write=100, cache_write_12h=100)
    priced = price_call(usage=usage, rates=rates(input=3, cacheWrite=3.75, output=15))
    assert "cache_write_12h_rate" in priced.missing


# --- AT-21 (UG-21) provider-reported wins, zero included -----------------------


def test_at21_reported_zero_is_zero_and_not_flagged():
    priced = price_call(usage=tokens(10_000, 1_000), rates=rates(input=3, output=20), reported_cost=0)
    assert (priced.cost_usd, priced.source, priced.pricing_missing) == (D(0), "provider_reported", False)


def test_at21_nothing_added_on_top_of_a_report():
    priced = price_call(
        usage=tokens(1000, 10, cache_read=300), rates=rates(input=3, cacheRead=0.3, output=15),
        reported_cost=0.02,
    )
    assert priced.cost_usd == D("0.02")


def test_ug21_a_negative_or_bad_report_is_ignored():
    for bad in (-1, float("nan"), True, "x"):
        priced = price_call(usage=tokens(1000, 0), rates=rates(input=3, output=15), reported_cost=bad)
        assert priced.source == "model_pricing"


# --- AT-22 (UG-22) units --------------------------------------------------------


def test_at22_per_image():
    model = Rates.from_mapping({"unit": "image", "unitPrice": 0.04})
    priced = price_call(usage=TokenUsage(), rates=model, units=Units(images=2))
    assert priced.cost_usd == D("0.08")
    assert priced.pricing_mode == "unit"


def test_at22_per_megapixel():
    model = Rates.from_mapping({"unit": "megapixel", "unitPrice": 0.01})
    priced = price_call(usage=TokenUsage(), rates=model, units=Units(image_sizes=[(1000, 1000), (2000, 1000)]))
    assert priced.cost_usd == D("0.03")


def test_at22_token_rates_and_a_request_price_are_both_charged():
    model = Rates.from_mapping({"inputPer1M": 3, "outputPer1M": 15, "unit": "request", "unitPrice": 0.001})
    priced = price_call(usage=tokens(1000, 0), rates=model)
    assert priced.cost_usd == D("0.004")


def test_at22_per_second_without_a_measure_is_flagged():
    model = Rates.from_mapping({"unit": "second", "unitPrice": 0.01})
    priced = price_call(usage=TokenUsage(), rates=model)
    assert priced.missing == ("secondQuantity",)
    assert priced.cost_usd == D(0) and priced.pricing_missing


def test_at22_images_with_no_price_are_flagged():
    priced = price_call(usage=TokenUsage(), rates=Rates(), units=Units(images=2))
    assert "outputImageTokens" in priced.missing


def test_ug22_unit_only_model_records_tokens_without_charging_them():
    model = Rates.from_mapping({"unit": "image", "unitPrice": 0.04})
    priced = price_call(usage=tokens(5000, 5000), rates=model, units=Units(images=1))
    assert priced.cost_usd == D("0.04")
    assert not priced.pricing_missing


# --- AT-23 (UG-23) aborted calls ---------------------------------------------


def test_at23_cut_off_is_estimated_at_the_model_rates():
    estimate = estimate_tokens(prompt_estimate=2000, max_output=None)
    assert estimate == (2000, 4096)
    priced = price_aborted(outcome=CUT_OFF, rates=rates(input=3, output=15), estimate=estimate)
    assert priced.cost_usd == D("0.06744")
    assert priced.source == "estimated_on_abort"
    assert quality(priced.cost_usd, priced.source, priced.pricing_missing) == "estimated"


def test_lm24_estimate_defaults():
    assert estimate_tokens() == (16_000, 4_096)
    assert estimate_tokens(last_prompt_size=900, max_output=256) == (900, 256)


def test_at23_not_contacted_costs_nothing():
    priced = price_aborted(outcome=NOT_CONTACTED, rates=rates(input=3, output=15))
    assert (priced.cost_usd, priced.provider_contacted, priced.pricing_missing) == (D(0), False, False)


def test_at23_refused_with_no_usage_is_free_and_contacted():
    priced = price_aborted(outcome=REFUSED, rates=rates(input=3, output=15))
    assert (priced.cost_usd, priced.source, priced.provider_contacted) == (D(0), "free", True)


def test_at23_neither_usage_nor_estimate_is_unknown_and_flagged():
    priced = price_aborted(outcome=NO_ESTIMATE, rates=rates(input=3, output=15))
    assert priced.source == "unknown" and priced.pricing_missing


# --- AT-28 (UG-28) interaction cost source -----------------------------------


@pytest.mark.parametrize(
    "entries, expected",
    [
        ([("provider_reported", "0.01"), ("model_pricing", "0.02")], "mixed"),
        ([("fallback_pricing", "0.02")], "fallback_pricing"),
        ([("fallback_pricing", "0.02"), ("model_pricing", "0.01")], "mixed"),
        ([("tool_pricing_rule", "0.014")], "tool_pricing_rule"),
        ([("provider_reported", "0"), ("free", "0")], "provider_reported"),
        ([("free", "0"), ("free", "0")], "free"),
        ([("unknown", "0")], "unknown"),
    ],
)
def test_at28_interaction_source(entries, expected):
    assert interaction_source(Settled(D(cost), source) for source, cost in entries) == expected


# --- AT-29 (UG-29) the split adds up -----------------------------------------


def _every_priced_call():
    model = rates(input=3, output=15)
    yield price_call(usage=tokens(1000, 100), rates=model, reported_cost=0.01)
    yield price_call(usage=tokens(6000, 400), rates=model)
    yield price_call(usage=tokens(1000, 500), rates=None)
    yield price_call(usage=tokens(1000, 500), rates=rates(input=0, output=0))
    yield price_call(usage=tokens(1000, 0, cache_read=1000), rates=model)
    yield price_call(usage=tokens(1000, 500), rates=model, service_tier="flex")
    yield price_call(usage=TokenUsage(), rates=Rates.from_mapping({"unit": "image", "unitPrice": 0.04}), units=Units(images=2))
    yield price_aborted(outcome=CUT_OFF, rates=model, estimate=(2000, 4096))
    yield price_aborted(outcome=REFUSED, rates=model)
    yield price_aborted(outcome=NO_ESTIMATE, rates=model)


def test_at29_parts_sum_to_the_cost_and_none_is_negative():
    entries = [Settled(p.cost_usd, p.source, p.pricing_missing) for p in _every_priced_call()]
    for entry in entries:
        parts = split(entry)
        assert abs(parts.total() - entry.cost_usd) <= TOLERANCE
        assert min(parts.provider, parts.calculated, parts.estimated, parts.unresolved) >= 0
    result = totals(entries)
    assert abs(result.split.total() - result.cost_usd) <= TOLERANCE


def test_at29_partial_free_contacted_counts_as_unresolved():
    entry = Settled(D(0), "free", status="partial", provider_contacted=True)
    assert is_unresolved(entry)
    assert not is_unresolved(Settled(D(0), "free", status="completed"))


def test_ug29_which_part_each_source_lands_in():
    assert split(Settled(D("0.01"), "provider_reported")) == CostSplit(provider=D("0.010000000000"))
    assert split(Settled(D("0.01"), "estimated_on_abort")).estimated == D("0.01")
    assert split(Settled(D("0.01"), "fallback_pricing")).unresolved == D("0.01")
    assert split(Settled(D("0.01"), "model_pricing", pricing_missing=True)).unresolved == D("0.01")
    assert split(Settled(D("0.01"), "model_pricing")).calculated == D("0.01")


def test_cost_quality_labels():
    assert quality(D(0), "fallback_pricing", True) == "unpriced"
    assert quality(D("0.1"), "fallback_pricing", True) == "estimated"
    assert quality(D("0.1"), "provider_reported", False) == "provider_reported"
    assert quality(D("0.1"), "model_pricing", False) == "calculated"


# --- AT-9 (UG-9) provider evidence is bounded --------------------------------


def test_at9_only_counters_survive():
    raw = {
        "prompt_tokens": 12,
        "output_text": "the reply, never stored",
        "authorization": "Bearer secret",
        "token_counts": list(range(41)),
        "cost_details": {"upstream_inference_cost": 0.002, "note": "x"},
        "currency": "USD",
    }
    cleaned = sanitise_provider_usage(raw)
    assert cleaned == {
        "prompt_tokens": 12,
        "token_counts": list(range(40)),
        "cost_details": {"upstream_inference_cost": 0.002},
        "currency": "USD",
    }
    assert "secret" not in repr(cleaned) and "reply" not in repr(cleaned)


def test_ug9_depth_and_value_limits():
    deep = {"usage": {"usage": {"usage": {"usage": {"usage": {"usage": {"usage": 1}}}}}}}
    assert "1" not in repr(sanitise_provider_usage(deep))
    many = {f"token_{i}": i for i in range(600)}
    assert len(sanitise_provider_usage(many)) == 500
    assert sanitise_provider_usage({"x" * 81 + "token": 1}) == {}
    assert sanitise_provider_usage("not a dict") == {}


def test_ug9_provider_ids():
    assert provider_id("gen-1731234567-AbC.def:1/2@x+y~z") is not None
    assert provider_id("-starts-with-dash") is None
    assert provider_id("has space") is None
    assert provider_id("a" * 257) is None
    assert provider_id(42) is None

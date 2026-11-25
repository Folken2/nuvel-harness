"""What one provider call cost, and how sure that number is.

The rules of usage-and-analytics.md 3.2 (UG-18 to UG-23), with no I/O, so the
ledger writer and the run budget share one answer. Amounts are ``Decimal``
rounded to 12 places: the worked example of AT-19 must come out exact.

Blank means unknown and 0 means free: a rate that is ``None`` is never priced
as 0 (UG-20).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping, Optional

from .tokens import TokenUsage

MILLION = Decimal(1_000_000)
PLACES = Decimal("0.000000000001")
ZERO = Decimal(0)

# When the catalogue has no positive rate at all (UG-20).
DEFAULT_FALLBACK_INPUT_PER_1M = Decimal(15)
DEFAULT_FALLBACK_OUTPUT_PER_1M = Decimal(60)

# LM-24: when a cut-off call's prompt size and maximum output are not known.
DEFAULT_ESTIMATE_INPUT_TOKENS = 16_000
DEFAULT_ESTIMATE_OUTPUT_TOKENS = 4_096

UNITS = ("image", "megapixel", "second", "request")
STANDARD_TIERS = (None, "", "default", "standard")

# Cost sources (section 2, "Cost sources").
PROVIDER_REPORTED = "provider_reported"
MODEL_PRICING = "model_pricing"
FALLBACK_PRICING = "fallback_pricing"
ESTIMATED_ON_ABORT = "estimated_on_abort"
FREE = "free"
UNKNOWN = "unknown"


def to_decimal(value: object) -> Optional[Decimal]:
    """A finite, non-negative number as a ``Decimal``; anything else is unknown."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() and value >= 0 else None
    if isinstance(value, (int, float)):
        return Decimal(str(value)) if math.isfinite(value) and value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except ArithmeticError:
            return None
        return parsed if parsed.is_finite() and parsed >= 0 else None
    return None


def money(value: Decimal) -> Decimal:
    """Stored to 12 places; negatives and non-finite values store as 0 (UG-19)."""
    if not value.is_finite() or value <= 0:
        return ZERO.quantize(PLACES)
    return value.quantize(PLACES, rounding=ROUND_HALF_UP)


# The rate names a snapshot carries, USD per million tokens.
RATE_FIELDS = (
    "input",
    "output",
    "cache_read",
    "cache_write",
    "cache_write_1h",
    "input_text",
    "input_image",
    "input_audio",
    "output_text",
    "output_image",
    "output_audio",
)


@dataclass(frozen=True)
class Rates:
    """One model's prices at the moment of the call. ``None`` is unknown."""

    input: Optional[Decimal] = None
    output: Optional[Decimal] = None
    cache_read: Optional[Decimal] = None
    cache_write: Optional[Decimal] = None
    cache_write_1h: Optional[Decimal] = None
    input_text: Optional[Decimal] = None
    input_image: Optional[Decimal] = None
    input_audio: Optional[Decimal] = None
    output_text: Optional[Decimal] = None
    output_image: Optional[Decimal] = None
    output_audio: Optional[Decimal] = None
    currency: str = "USD"
    unit: Optional[str] = None
    unit_price: Optional[Decimal] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Rates":
        """From a snapshot: ``<name>Per1M`` per million, or ``<name>PerToken``.

        Names are camel case (``cacheReadPer1M``). The per-million rate wins
        when both are present.
        """

        def rate(name: str) -> Optional[Decimal]:
            camel = "".join(
                part if i == 0 else part.capitalize() for i, part in enumerate(name.split("_"))
            )
            per_million = to_decimal(raw.get(f"{camel}Per1M"))
            if per_million is not None:
                return per_million
            per_token = to_decimal(raw.get(f"{camel}PerToken"))
            return per_token * MILLION if per_token is not None else None

        unit = raw.get("unit")
        currency = raw.get("currency")
        return cls(
            **{name: rate(name) for name in RATE_FIELDS},
            currency=currency.upper() if isinstance(currency, str) and currency else "USD",
            unit=unit if unit in UNITS else None,
            unit_price=to_decimal(raw.get("unitPrice")),
        )

    def snapshot(self) -> dict[str, Any]:
        """The rates used, copied onto the entry so a later price change can't rewrite it."""
        out: dict[str, Any] = {
            name: str(value)
            for name in RATE_FIELDS
            if (value := getattr(self, name)) is not None
        }
        out["currency"] = self.currency
        if self.unit:
            out["unit"] = self.unit
        if self.unit_price is not None:
            out["unitPrice"] = str(self.unit_price)
        return out


@dataclass(frozen=True)
class FallbackRates:
    """The conservative rates of UG-20, used where a needed base rate is missing."""

    input: Decimal = DEFAULT_FALLBACK_INPUT_PER_1M
    output: Decimal = DEFAULT_FALLBACK_OUTPUT_PER_1M

    @classmethod
    def from_catalogue(
        cls, inputs: list[object], outputs: list[object]
    ) -> "FallbackRates":
        """The highest positive input and output rates of any catalogue model."""

        def highest(values: list[object], default: Decimal) -> Decimal:
            positive = [v for v in (to_decimal(x) for x in values) if v is not None and v > 0]
            return max(positive) if positive else default

        return cls(
            input=highest(inputs, DEFAULT_FALLBACK_INPUT_PER_1M),
            output=highest(outputs, DEFAULT_FALLBACK_OUTPUT_PER_1M),
        )


@dataclass(frozen=True)
class Units:
    """Measured quantities for unit pricing (UG-22). ``None`` is not measured."""

    images: Optional[int] = None
    # Width and height of each returned image, in pixels.
    image_sizes: Optional[list[tuple[int, int]]] = None
    # Set when the sizes came from the request parameters, not the result.
    sizes_from_request: bool = False
    seconds: Optional[Decimal] = None


@dataclass(frozen=True)
class PricedCall:
    """One invocation's cost, with the evidence the ledger keeps (section 2, "Details")."""

    cost_usd: Decimal
    source: str
    pricing_missing: bool
    # The meters that lacked a price, named as in section 2.
    missing: tuple[str, ...] = ()
    breakdown: dict[str, Decimal] = field(default_factory=dict)
    snapshot: dict[str, Any] = field(default_factory=dict)
    pricing_mode: str = "tokens"
    provider_contacted: bool = True
    billing_units: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Meter:
    """Running total while a call is priced."""

    fallback: FallbackRates
    missing: list[str] = field(default_factory=list)
    used_fallback: bool = False

    def flag(self, meter: str) -> None:
        if meter not in self.missing:
            self.missing.append(meter)

    def base(self, rate: Optional[Decimal], tokens: int, meter: str, conservative: Decimal) -> Decimal:
        """Tokens at a base rate, or at the conservative one when it is missing."""
        if tokens <= 0:
            return ZERO
        if rate is None:
            self.flag(meter)
            self.used_fallback = True
            rate = conservative
        return tokens * rate

    def derived(self, tokens: int, meter: str, *rates: Optional[Decimal]) -> Decimal:
        """Tokens at the first known rate. A fallback past the first is flagged."""
        if tokens <= 0:
            return ZERO
        for i, rate in enumerate(rates):
            if rate is None:
                continue
            if i > 0:
                self.flag(meter)
            return tokens * rate
        self.flag(meter)
        return ZERO


def _token_cost(usage: TokenUsage, rates: Rates, meter: _Meter) -> dict[str, Decimal]:
    """The formula of UG-19, per million tokens, before rounding."""
    fallback = meter.fallback
    input_rate = rates.input
    # A missing base rate is replaced by the conservative one for every bucket.
    effective_input = input_rate if input_rate is not None else fallback.input

    image_in = usage.input_image
    audio_in = usage.input_audio
    text_in = max(0, usage.ordinary_input - image_in - audio_in)
    input_cost = meter.base(
        rates.input_text if rates.input_text is not None else input_rate,
        text_in,
        "inputTokens",
        fallback.input,
    )
    input_cost += meter.derived(image_in, "inputImageTokens", rates.input_image, effective_input)
    input_cost += meter.derived(audio_in, "inputAudioTokens", rates.input_audio, effective_input)

    # An input rate of 0 with a blank cache rate: the cache part is 0, not flagged (MC-38).
    zero_input = input_rate is not None and input_rate == 0
    cached = (
        ZERO
        if zero_input and rates.cache_read is None
        else meter.derived(usage.cache_read, "cacheReadTokens", rates.cache_read, effective_input)
    )
    plain_write = usage.cache_write - usage.cache_write_1h
    write = (
        ZERO
        if zero_input and rates.cache_write is None
        else meter.derived(plain_write, "cacheWriteTokens", rates.cache_write, effective_input)
    )
    write_1h = (
        ZERO
        if zero_input and rates.cache_write_1h is None and rates.cache_write is None
        else meter.derived(
            usage.cache_write_1h,
            "cacheWrite1hTokens",
            rates.cache_write_1h,
            rates.cache_write,
            effective_input,
        )
    )

    output_rate = rates.output
    effective_output = output_rate if output_rate is not None else fallback.output
    image_out = usage.output_image
    audio_out = usage.output_audio
    text_out = max(0, usage.output_total - image_out - audio_out)
    output_cost = meter.base(
        rates.output_text if rates.output_text is not None else output_rate,
        text_out,
        "outputTokens",
        fallback.output,
    )
    output_cost += meter.derived(image_out, "outputImageTokens", rates.output_image, effective_output)
    output_cost += meter.derived(audio_out, "outputAudioTokens", rates.output_audio, effective_output)

    return {
        "input": input_cost / MILLION,
        "cached_input": cached / MILLION,
        "cache_write": write / MILLION,
        "cache_write_1h": write_1h / MILLION,
        "output": output_cost / MILLION,
    }


def _unit_cost(
    rates: Rates, units: Optional[Units], usage: TokenUsage, token_priced: bool, meter: _Meter
) -> tuple[Decimal, dict[str, Any]]:
    """Units of UG-22: requests are counted, everything else must be measured."""
    unit, price = rates.unit, rates.unit_price
    units = units or Units()
    if unit is None or price is None:
        # Images came back and nothing can price them: flagged, not free.
        if (units.images or units.image_sizes) and not token_priced:
            meter.flag("outputImageTokens")
        return ZERO, {}
    if unit == "request":
        return price, {"request": 1}
    if unit == "second":
        if units.seconds is None:
            meter.flag("secondQuantity")
            return ZERO, {}
        return units.seconds * price, {"second": str(units.seconds)}
    # Images: tokens win when they can price them, but 0 tokens never hide a charge.
    if token_priced and usage.output_image > 0:
        return ZERO, {}
    if unit == "image":
        if units.images is None:
            meter.flag("imageQuantity")
            return ZERO, {}
        return units.images * price, {"image": units.images}
    sizes = units.image_sizes
    if not sizes:
        meter.flag("megapixelQuantity")
        return ZERO, {}
    if units.sizes_from_request:
        meter.flag("megapixelQuantity")
    megapixels = sum(Decimal(w) * Decimal(h) for w, h in sizes) / MILLION
    return megapixels * price, {"megapixel": str(megapixels)}


def _has_token_rates(rates: Rates) -> bool:
    return any(getattr(rates, name) is not None for name in RATE_FIELDS)


def price_call(
    *,
    usage: TokenUsage,
    rates: Optional[Rates],
    fallback: FallbackRates = FallbackRates(),
    reported_cost: object = None,
    units: Optional[Units] = None,
    service_tier: Optional[str] = None,
) -> PricedCall:
    """A call the provider answered, with its usage (UG-18 to UG-22).

    ``reported_cost`` is what the provider stated for this call, 0 included.
    ``rates`` are the model's own rates; Auto and unknown models pass ``None``.
    ``service_tier`` is the tier the provider reported, never the one requested.
    """
    rates = rates or Rates()
    snapshot = rates.snapshot()
    reported = to_decimal(reported_cost)
    if reported is not None:
        # The provider's total is complete: no cache or search fee goes on top (UG-21).
        return PricedCall(
            cost_usd=money(reported),
            source=PROVIDER_REPORTED,
            pricing_missing=usage.incomplete,
            missing=(),
            breakdown={},
            snapshot=snapshot,
            pricing_mode="none",
        )

    meter = _Meter(fallback=fallback)
    if rates.currency != "USD":
        # Not comparable with USD costs: priced as if every rate were missing.
        meter.flag("tokenCurrency")
        rates = Rates(unit=rates.unit, unit_price=rates.unit_price)

    unit_only = rates.unit is not None and not _has_token_rates(rates)
    breakdown = {} if unit_only else _token_cost(usage, rates, meter)
    token_priced = not unit_only and rates.input is not None and rates.output is not None
    unit_cost, billing_units = _unit_cost(rates, units, usage, token_priced, meter)
    breakdown["unit"] = unit_cost

    if service_tier not in STANDARD_TIERS:
        meter.flag("non_standard_service_tier")
    if usage.cache_write_12h > 0:
        meter.flag("cache_write_12h_rate")

    cost = money(sum(breakdown.values(), ZERO))
    missing = tuple(meter.missing)
    # Last-step-only figures are priced but never trusted as complete (UG-17).
    pricing_missing = bool(missing) or usage.incomplete
    if meter.used_fallback:
        source = FALLBACK_PRICING
    elif cost == 0 and not pricing_missing and _all_free(rates, usage):
        source = FREE
    else:
        source = MODEL_PRICING
    return PricedCall(
        cost_usd=cost,
        source=source,
        pricing_missing=pricing_missing,
        missing=missing,
        breakdown={k: money(v) for k, v in breakdown.items()},
        snapshot=snapshot,
        pricing_mode="unit" if unit_only else "tokens",
        billing_units=billing_units,
    )


def _all_free(rates: Rates, usage: TokenUsage) -> bool:
    """Every rate the call needed is known and explicitly 0."""
    needed = []
    if usage.ordinary_input or usage.cache_read or usage.cache_write:
        needed.append(rates.input)
    if usage.output_total:
        needed.append(rates.output)
    if rates.unit is not None:
        needed.append(rates.unit_price)
    return bool(needed) and all(rate is not None and rate == 0 for rate in needed)


# How a call that did not finish normally ended (UG-23).
NOT_CONTACTED = "not_contacted"
REFUSED = "refused"
CUT_OFF = "cut_off"
NO_ESTIMATE = "no_estimate"


def estimate_tokens(
    prompt_estimate: Optional[int] = None,
    last_prompt_size: Optional[int] = None,
    max_output: Optional[int] = None,
) -> tuple[int, int]:
    """Input and output a cut-off call is charged for (limits.md LM-24)."""
    input_tokens = next(
        (n for n in (prompt_estimate, last_prompt_size) if isinstance(n, int) and n > 0),
        DEFAULT_ESTIMATE_INPUT_TOKENS,
    )
    output_tokens = (
        max_output if isinstance(max_output, int) and max_output > 0 else DEFAULT_ESTIMATE_OUTPUT_TOKENS
    )
    return input_tokens, output_tokens


def price_aborted(
    *,
    outcome: str,
    rates: Optional[Rates],
    fallback: FallbackRates = FallbackRates(),
    estimate: Optional[tuple[int, int]] = None,
) -> PricedCall:
    """A call cancelled or failed with no usage figures (UG-23).

    A call that did return usage is priced by ``price_call`` with its final
    status instead. ``estimate`` is the (input, output) of ``estimate_tokens``,
    stored when the call started.
    """
    rates = rates or Rates()
    if outcome == NOT_CONTACTED:
        return PricedCall(
            cost_usd=money(ZERO), source=FREE, pricing_missing=False,
            snapshot=rates.snapshot(), pricing_mode="none", provider_contacted=False,
        )
    if outcome == REFUSED:
        return PricedCall(
            cost_usd=money(ZERO), source=FREE, pricing_missing=False,
            snapshot=rates.snapshot(), pricing_mode="none",
        )
    if outcome == CUT_OFF and estimate is not None:
        input_tokens, output_tokens = estimate
        priced = price_call(
            usage=TokenUsage(
                input_total=input_tokens,
                ordinary_input=input_tokens,
                output_total=output_tokens,
                total=input_tokens + output_tokens,
            ),
            rates=rates,
            fallback=fallback,
        )
        return PricedCall(
            cost_usd=priced.cost_usd,
            source=ESTIMATED_ON_ABORT,
            pricing_missing=priced.pricing_missing,
            missing=priced.missing,
            breakdown=priced.breakdown,
            snapshot=priced.snapshot,
        )
    # Contacted, with neither usage nor an estimate: unknown, never free.
    return PricedCall(
        cost_usd=money(ZERO), source=UNKNOWN, pricing_missing=True,
        snapshot=rates.snapshot(), pricing_mode="none",
    )

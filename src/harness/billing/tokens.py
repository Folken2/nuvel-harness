"""One model call's token counts, put into the ledger's buckets.

Providers disagree on whether input includes cached tokens and whether output
includes reasoning. ``normalise`` turns what one reported into the buckets of
usage-and-analytics.md section 2 ("Token buckets") and UG-17, so pricing and
reports never have to know which provider it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenUsage:
    """Every count is a whole number, never negative.

    ``input_total`` includes the cache buckets and ``output_total`` includes
    reasoning: those two are the ledger's ``tokens in`` and ``tokens out``. The
    cache write lifetimes and the modality counts are subsets, never added again.
    """

    input_total: int = 0
    ordinary_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0
    cache_write_30m: int = 0
    cache_write_12h: int = 0
    output_total: int = 0
    reasoning: int = 0
    total: int = 0
    # Subsets of input and output total, where the provider reported them.
    input_image: int = 0
    input_audio: int = 0
    output_image: int = 0
    output_audio: int = 0
    # The figures cover only part of the run (UG-17), so the entry is flagged.
    incomplete: bool = False


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if value != value or value in (float("inf"), float("-inf")):
        return 0
    return max(0, int(value))


def normalise(
    *,
    input_tokens: object = 0,
    input_includes_cache: bool = True,
    cache_read: object = 0,
    cache_write: object = 0,
    cache_write_1h: object = 0,
    cache_write_30m: object = 0,
    cache_write_12h: object = 0,
    output_tokens: object = 0,
    reasoning: object = 0,
    output_includes_reasoning: bool = True,
    total: object = 0,
    input_image: object = 0,
    input_audio: object = 0,
    output_image: object = 0,
    output_audio: object = 0,
    incomplete: bool = False,
) -> TokenUsage:
    """The buckets for what a provider reported (UG-17).

    ``input_includes_cache`` says whether ``input_tokens`` already counts the
    cache read and write tokens; ``output_includes_reasoning`` whether
    ``output_tokens`` already counts ``reasoning``. Set ``incomplete`` when the
    figures are those of the last step of a multi-step call only.
    """
    read = _count(cache_read)
    write = _count(cache_write)
    reported_input = _count(input_tokens)
    input_total = reported_input if input_includes_cache else reported_input + read + write
    # Derived from the buckets when the provider left the total at 0.
    input_total = max(input_total, read + write) if input_total == 0 else input_total
    read = min(read, input_total)
    write = min(write, input_total - read)
    # Lifetimes are capped in the order 1 hour, 30 minutes, 12 hours.
    write_1h = min(_count(cache_write_1h), write)
    write_30m = min(_count(cache_write_30m), write - write_1h)
    write_12h = min(_count(cache_write_12h), write - write_1h - write_30m)

    thoughts = _count(reasoning)
    output_total = _count(output_tokens)
    if not output_includes_reasoning:
        output_total += thoughts
    thoughts = min(thoughts, output_total)

    reported_total = _count(total)
    image_in = min(_count(input_image), input_total)
    image_out = min(_count(output_image), output_total)
    return TokenUsage(
        input_total=input_total,
        ordinary_input=max(0, input_total - read - write),
        cache_read=read,
        cache_write=write,
        cache_write_1h=write_1h,
        cache_write_30m=write_30m,
        cache_write_12h=write_12h,
        output_total=output_total,
        reasoning=thoughts,
        total=reported_total if reported_total > 0 else input_total + output_total,
        input_image=image_in,
        input_audio=min(_count(input_audio), input_total - image_in),
        output_image=image_out,
        output_audio=min(_count(output_audio), output_total - image_out),
        incomplete=incomplete,
    )


def summed(calls: list[TokenUsage]) -> Optional[TokenUsage]:
    """Every step of a multi-step call added together (UG-17); None for no steps."""
    if not calls:
        return None
    fields = [f for f in TokenUsage.__dataclass_fields__ if f != "incomplete"]
    totals = {f: sum(getattr(call, f) for call in calls) for f in fields}
    return TokenUsage(**totals, incomplete=any(call.incomplete for call in calls))

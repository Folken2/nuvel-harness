"""The part of one call's cost above the person's month limit (limits.md LM-22). Pure.

Overage is a record for admins, never a charge and never a stop: a run that has
started finishes (LM-20), and each of its calls past the limit says by how much.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

__all__ = ["overage"]

ZERO = Decimal(0)


def overage(before: Decimal, cost: Decimal, limit: Optional[Decimal]) -> Decimal:
    """Month cost before the call ``before``, the call's ``cost``, the run's ``limit``.

    No limit (an unlimited person) has no overage. Otherwise the part of ``cost`` that
    lands above ``limit``: none while under it, all of it once the month is already there.
    """
    if limit is None or cost <= ZERO:
        return ZERO
    if before >= limit:
        return cost
    return max(ZERO, before + cost - limit)

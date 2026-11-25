"""What ``find_tools`` loads for a query. Pure.

Two forms:
- ``select:name_a,name_b``: those tools, by exact name.
- Keywords: the catalog ranked by them, best first, at most ``limit``. A word marked
  ``+word`` must appear in every match. A word in a tool's name counts more than one in
  its description, and a word may be part of a longer one ("calendar" in
  "GOOGLECALENDAR_CREATE_EVENT").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

SELECT = "select:"
REQUIRED = "+"
# A word in the name, then one in the description.
_IN_NAME = 3
_IN_DESCRIPTION = 1


@dataclass(frozen=True)
class Selected:
    found: tuple[str, ...]
    # Names asked for that the catalog doesn't have.
    not_found: tuple[str, ...]


def is_select(query: str) -> bool:
    return query.strip().lower().startswith(SELECT)


def select(query: str, catalog: Mapping[str, str]) -> Selected:
    """The tools a ``select:`` query names, in the order asked, each once."""
    asked = [name.strip() for name in query.strip()[len(SELECT) :].split(",") if name.strip()]
    found: list[str] = []
    not_found: list[str] = []
    for name in dict.fromkeys(asked):
        (found if name in catalog else not_found).append(name)
    return Selected(tuple(found), tuple(not_found))


def _score(words: list[str], name: str, description: str) -> int:
    return sum(
        _IN_NAME if word in name else _IN_DESCRIPTION if word in description else 0 for word in words
    )


def rank(query: str, catalog: Mapping[str, str], limit: int) -> tuple[str, ...]:
    """The best matches for keywords, by score and then name."""
    terms = [term.lower() for term in query.split() if term.strip(REQUIRED)]
    required = [term[1:] for term in terms if term.startswith(REQUIRED)]
    words = [term.lstrip(REQUIRED) for term in terms]
    scored = []
    for name, description in catalog.items():
        lowered_name, lowered_description = name.lower(), description.lower()
        if any(word not in lowered_name and word not in lowered_description for word in required):
            continue
        score = _score(words, lowered_name, lowered_description)
        if score > 0:
            scored.append((-score, name))
    return tuple(name for _, name in sorted(scored)[:limit])

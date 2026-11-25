"""Whether a run keeps files of its conversation.

A conversation's run does, when files are on. A run that is no conversation of the
person's (a background task, in its own session) keeps none. When files are on and
which runs are a conversation's is the product's to say: it sets the rule
(``app_files.py`` is this service's). With no rule set, no run keeps files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FilesRule:
    # Whether files are on. Asked at each call, so a setting is the one in force then.
    enabled: Callable[[], bool]
    # The run's state -> whether this run is a conversation's.
    in_conversation: Callable[[Any], bool]


_KEEPS_NONE = FilesRule(enabled=lambda: False, in_conversation=lambda state: False)
_rule: FilesRule = _KEEPS_NONE


def set_rule(rule: FilesRule) -> None:
    global _rule
    _rule = rule


def keeps_files(state: Any) -> bool:
    return _rule.enabled() and _rule.in_conversation(state)

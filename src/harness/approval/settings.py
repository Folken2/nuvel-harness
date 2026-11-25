"""A turn's approval settings: the deployment's mode and the tools this person always allows.

The default is `smart`, with nothing allowed. How a product's turn carries them is its
own (web_app_settings.py reads this service's).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["smart", "strict", "off"]
MODES: tuple[Mode, ...] = ("smart", "strict", "off")


@dataclass(frozen=True)
class ApprovalSettings:
    # smart: ask for what has effects, and the classifier rates what isn't clear.
    # strict: ask for everything that isn't plainly read only. off: never ask.
    mode: Mode = "smart"
    allowed: frozenset[str] = frozenset()


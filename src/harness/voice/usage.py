"""What a speech call cost, for the web app's usage ledger."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class AudioUsage:
    kind: Literal["speech_to_text", "text_to_speech"]
    model: str
    cost_usd: float
    # Seconds heard for speech to text, characters spoken for text to speech.
    amount: float


UsageSink = Callable[[AudioUsage], None]


@dataclass
class UsageTally:
    """Adds up the calls one request made."""

    calls: list[AudioUsage] = field(default_factory=list)

    def __call__(self, usage: AudioUsage) -> None:
        self.calls.append(usage)

    @property
    def cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    @property
    def amount(self) -> float:
        return sum(call.amount for call in self.calls)

    @property
    def model(self) -> Optional[str]:
        return self.calls[-1].model if self.calls else None

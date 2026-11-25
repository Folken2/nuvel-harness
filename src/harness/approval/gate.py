"""Before a tool runs: let it run, or pause the run until the person approves it.

The pause is ADK's tool confirmation: `request_confirmation` on the call, and ADK
asks the person with an `adk_request_confirmation` call that waits like a question
(tools/ask.py). Their answer comes back through here with `tool_confirmation` set:
approved runs the call, once, and denied answers it as denied, which the model reads
as the person's decision rather than an error. A call ADK brings back is never
classified again, so the person is never asked twice for one call.

A classifier that fails or times out counts as risky: the call asks.

After ``max_denials`` refusals in a row the gate stops asking for the rest of the turn
(breaker.py): the model is told to make no more calls and to talk to the person.

Given an ``authorisation_threshold``, the gate asks a second question apart from the first:
did the person ask for this call, in their own messages (authorisation.py)? It never lets
a call run that the first would ask for. Every card carries its answer, so a call nobody
asked for stands out. And a call the first rates safe asks after all when the run has read
outside content since the person last wrote and the person plainly didn't ask for it,
since that content may be what steered the model to it. A failed check there asks too.

The gate is built with everything that is a product's: how a call reads as actions, the
policy, the classifier, where an action's hints come from and whose key the classifier
call goes on (``approval/product_rules.py`` builds this service's).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Optional

from google.adk.labs.classifiers import BaseClassifier, Binary, BinaryAnswer
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .. import wire_names
from .actions import AGENT, Action, ActionReader
from .authorisation import AUTHORISED, authorisation_state, person_messages
from .breaker import STOPPED_NOTE, denials_in_a_row, stopped
from .other_waits import not_decided
from .policy import TOO_LONG_REASON, UNCHECKED_REASON, Policy, Verdict
from .provenance import read_outside_since_person
from .settings import ApprovalSettings

logger = logging.getLogger(__name__)

QUESTION = Binary(
    instructions=(
        "Would running this tool call have an effect outside the chat: create, change"
        " or delete data anywhere, send or post something, contact someone, or spend"
        " money?"
    ),
    criteria={
        "true": (
            "It writes, sends, deletes, moves, books, pays, or runs code that touches"
            " files, accounts or the network beyond computing an answer."
        ),
        "false": (
            "It only reads, searches, lists, fetches or computes, and changes nothing"
            " anyone else can see."
        ),
    },
)

NEEDS_APPROVAL = {
    "status": "needs_approval",
    "note": "Waiting for the person to approve or deny this call. It hasn't run.",
}
# A background task (runs/tasks.py) has nobody to show a card to, so a call that would
# ask is denied instead of pausing a run nobody can answer.
IN_TASK_NOTE = (
    "This call needs the person's approval, and a background task can't ask, so it"
    " didn't run. Say in your report what it would have done, so the turn that reads"
    " the report can do it with the person."
)
# Names their choice as the reason, even beside another: told only that the person denied
# it, a model gave a reason of its own, such as the sandbox having no network, which its
# instruction says and which would have stopped that call too.
DENIED_NOTE = (
    "The person chose not to allow this call, so it didn't run. Say that it didn't run"
    " because they didn't allow it: that is the reason, even if the call would have"
    " failed for another reason too, so don't give any other reason for why it didn't"
    " run. Don't try it again another way unless they ask. Tell them briefly what it"
    " would have done, and go on without it."
)
# The person answered another wait of the same step (./other_waits.py): not a refusal.
NOT_DECIDED_NOTE = (
    "The person answered your question and didn't decide on this call, so it didn't"
    " run. If it's still needed, make the call again and they'll be asked."
)
NOT_ASKED_FOR_REASON = "Not something you asked for, after reading content from outside this chat"
_ARG_TEXT_LIMIT = 2000
# Past this the call asks rather than have part of it rated: about 6k tokens, $0.0003.
CLASSIFIER_INPUT_LIMIT = 24_000

# The actions of a call -> what each one's app says it does, by the action's key.
HintsFor = Callable[[list[Action]], Awaitable[dict[str, frozenset[str]]]]
# The key the classifier's call goes on -> the classifier.
ClassifierFor = Callable[[str], BaseClassifier]
# The turn's state -> that key.
ApiKeyFor = Callable[[Any], str]
# The turn's approval settings, read from its state the product's way.
SettingsOf = Callable[[Any], ApprovalSettings]


def _clipped(value: Any) -> Any:
    """Arguments as the person sees them: long text cut, so a card stays a card."""
    if isinstance(value, str) and len(value) > _ARG_TEXT_LIMIT:
        return value[:_ARG_TEXT_LIMIT] + "…"
    if isinstance(value, dict):
        return {key: _clipped(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clipped(item) for item in value]
    return value


def _shown(action: Action) -> dict[str, Any]:
    """An action for the card. `clipped` says the card shows its arguments in part: what
    runs is the whole call, which the card can't hold."""
    args = _clipped(action.args)
    return {"key": action.key, "args": args, **({"clipped": True} if args != action.args else {})}


def denied_result(said: Optional[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "denied", "note": DENIED_NOTE}
    if said:
        result["person_said"] = said
    return result


def _said(payload: Any) -> Optional[str]:
    """The person's own words when they answered with a message rather than Deny."""
    text = payload.get("said") if isinstance(payload, dict) else None
    return text.strip() if isinstance(text, str) and text.strip() else None


def too_long(actions: list[Action]) -> bool:
    return len(json.dumps([action.args for action in actions], default=str)) > CLASSIFIER_INPUT_LIMIT


def classifier_state(tool_key: str, description: str, actions: list[Action]) -> str:
    """What the classifier rates: the call as data, kept apart from the question. Only
    the agent's own tools bring their description: a connector's is its author's text,
    written to be read by a model, and could argue for its own call."""
    own = all(action.source == AGENT for action in actions)
    return json.dumps(
        {
            "tool": tool_key,
            **({"description": description[:600]} if own and description else {}),
            "actions": [{"tool": action.key, "arguments": action.args} for action in actions],
        },
        default=str,
    )


# The product's own reason to ask for a call the policy lets run; None when it has none.
AsksAlso = Callable[[list[Action], ToolContext], Optional[Verdict]]


class ApprovalGate:
    def __init__(
        self,
        *,
        threshold: float,
        reader: ActionReader,
        policy: Policy,
        classifier_for: ClassifierFor,
        hints_for: HintsFor,
        api_key_for: ApiKeyFor,
        settings_of: SettingsOf,
        max_denials: int = 0,
        asks_also: Optional[AsksAlso] = None,
        authorisation_threshold: Optional[float] = None,
        inside: Iterable[str] = (),
    ) -> None:
        """``asks_also`` is the product's own reason to ask for a call the policy would
        run, from what the run has done so far (its session's events, its state): asked
        after the policy, and never in mode off or for a call the person always allows.

        ``authorisation_threshold`` turns on the second question: at or above it the person
        asked for the call. None asks only the first. ``inside`` names the tools whose
        answers carry nothing a third party wrote (provenance.py)."""
        self.threshold = threshold
        self.authorisation_threshold = authorisation_threshold
        self._inside = frozenset(inside)
        # After this many refusals in a row no more calls are put to the person; 0: no limit.
        self.max_denials = max_denials
        self._reader = reader
        self._policy = policy
        self._classifier_for = classifier_for
        self._hints_for = hints_for
        self._api_key_for = api_key_for
        self._settings_of = settings_of
        self._asks_also = asks_also

    async def _hints(self, tool: BaseTool, actions: list[Action]) -> dict[str, frozenset[str]]:
        hints = {self._reader.key_of(tool.name): getattr(tool, "effect_hints", frozenset())}
        try:
            hints.update(await self._hints_for(actions))
        except Exception:  # noqa: BLE001 - without hints the words and classifier decide
            logger.exception("tool hints failed")
        return hints

    async def _probability(
        self, name: str, question: Binary, state: str, tool: BaseTool, tool_context: ToolContext
    ) -> Optional[float]:
        """The classifier's probability of yes to ``question`` about ``state``; None when it
        failed, which never reads as an answer either way."""
        api_key = self._api_key_for(tool_context.state)
        try:
            response = await self._classifier_for(api_key).classify(input=state, questions={name: question})
            answer = response.answers[name]
            if not isinstance(answer, BinaryAnswer):
                raise ValueError(f"expected a yes/no answer, got {answer.type}")
            return answer.probability
        except Exception as error:  # noqa: BLE001 - a failed check asks the person
            logger.warning("approval classifier (%s) failed for %s: %s", name, tool.name, error)
            return None

    async def _risk(self, tool: BaseTool, actions: list[Action], tool_context: ToolContext) -> Optional[float]:
        """The probability that the call has effects. The call goes as data, apart from the
        question, and whole or not at all."""
        state = classifier_state(self._reader.key_of(tool.name), tool.description or "", actions)
        return await self._probability("effects", QUESTION, state, tool, tool_context)

    async def _authorised(self, tool: BaseTool, actions: list[Action], tool_context: ToolContext) -> Optional[float]:
        """The probability that the person asked for the call, in their own messages."""
        state = authorisation_state(person_messages(tool_context.session.events), actions)
        return await self._probability("authorised", AUTHORISED, state, tool, tool_context)

    @staticmethod
    def _denials(tool_context: ToolContext) -> int:
        return denials_in_a_row(tool_context.session.events, tool_context.invocation_id)

    async def check(
        self, tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        """None lets the call run; a dict answers it in the tool's place."""
        confirmation = tool_context.tool_confirmation
        if confirmation is not None:
            if confirmation.confirmed:
                logger.info("approval: %s approved", tool.name)
                return None
            if not_decided(confirmation.payload):
                logger.info("approval: %s not decided", tool.name)
                return {"status": "not_approved", "note": NOT_DECIDED_NOTE}
            logger.info("approval: %s denied", tool.name)
            denied = denied_result(_said(confirmation.payload))
            if stopped(self._denials(tool_context), self.max_denials):
                logger.warning("approval: %d calls denied in a row; no more are asked this turn", self.max_denials)
                denied["note"] = STOPPED_NOTE
            return denied

        actions = self._reader.actions_of(tool.name, args)
        settings = self._settings_of(tool_context.state)
        verdict = self._policy.decide(actions, await self._hints(tool, actions), settings)
        if verdict.kind == "run" and self._asks_also is not None and settings.mode != "off":
            always = all(action.readable and action.key in settings.allowed for action in actions)
            verdict = (None if always else self._asks_also(actions, tool_context)) or verdict
        risk: Optional[float] = None
        authorised: Optional[float] = None
        rated = False
        if verdict.kind == "classify" and too_long(actions):
            verdict = Verdict("ask", TOO_LONG_REASON)
        if verdict.kind == "classify":
            # The classifier only advises on what its input can steer.
            verdict = next((v for v in map(self._policy.second_check, actions) if v), verdict)
        if verdict.kind == "classify":
            risk = await self._risk(tool, actions, tool_context)
            if risk is None:
                verdict = Verdict("ask", UNCHECKED_REASON)
            else:
                verdict = Verdict("run" if risk < self.threshold else "ask", verdict.reason)
            needed = self.authorisation_threshold
            if (
                verdict.kind == "run"
                and needed is not None
                and read_outside_since_person(tool_context.session.events, self._inside)
            ):
                authorised, rated = await self._authorised(tool, actions, tool_context), True
                if authorised is None:
                    verdict = Verdict("ask", UNCHECKED_REASON)
                elif authorised < needed:
                    verdict = Verdict("ask", NOT_ASKED_FOR_REASON)
        logger.info(
            "approval: %s %s%s%s",
            tool.name,
            verdict.kind,
            "" if risk is None else f" risk={risk:.2f}",
            "" if authorised is None else f" authorised={authorised:.2f}",
        )
        if verdict.kind == "run":
            return None
        if tool_context.state.get(wire_names.TASK_MARK):
            logger.info("approval: %s denied in a background task", tool.name)
            return {"status": "denied", "note": IN_TASK_NOTE}

        if stopped(self._denials(tool_context), self.max_denials):
            logger.info("approval: %s not asked, the person denied %d in a row", tool.name, self.max_denials)
            return {"status": "denied", "note": STOPPED_NOTE}

        payload: dict[str, Any] = {
            "reason": verdict.reason,
            "risk": risk,
            "actions": [_shown(action) for action in actions],
        }
        if self.authorisation_threshold is not None:
            # Whether the person asked for it, for the card: None when unknown. A call too
            # long to rate whole isn't rated in part.
            if not rated and not too_long(actions):
                authorised = await self._authorised(tool, actions, tool_context)
            payload["authorised"] = authorised
            payload["asked_for"] = None if authorised is None else authorised >= self.authorisation_threshold
        tool_context.request_confirmation(hint=verdict.reason, payload=payload)
        # The pause isn't a result for the model to read: without this the model
        # is called again and makes the same call.
        tool_context.actions.skip_summarization = True
        return dict(NEEDS_APPROVAL)

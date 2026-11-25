"""Which calls ask the person first, in layers, before any classifier.

1. Never: the agent's own tools that only read or plan, and tools their app marks
   read only (a person's own server only for a name that changes nothing).
2. Always: actions that send, delete or pay, by the app's hint or by their name.
3. Everything else goes to the classifier (`smart`), or asks (`strict`).

A person's "always allow" skips the question for a call whose actions they all
allowed. Pure: no network, so every rule is tested on its own.

Which tools change nothing, whose hints are trusted and which calls get a second check
are data a ``Policy`` is built with: this module names no tool and no vendor.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Mapping, Optional

from .actions import SERVER, Action, own_name
from .settings import ApprovalSettings

Kind = Literal["run", "ask", "classify"]

# Verbs in an action's name that always ask, by what they do, unless a read verb
# comes first (GMAIL_LIST_SEND_AS lists aliases). Nouns that name the same things
# (POST, PAYMENT) ask only when nothing in the name reads.
_SENDS = frozenset({"SEND", "SENDS", "REPLY", "FORWARD", "PUBLISH", "INVITE", "TWEET"})
_DELETES = frozenset({"DELETE", "DELETES", "REMOVE", "TRASH", "DESTROY", "PURGE", "ERASE", "EMPTY", "WIPE"})
_PAYS = frozenset({"PAY", "PURCHASE", "BUY"})
_SEND_NOUNS = frozenset({"POST", "SHARE"})
_PAY_NOUNS = frozenset({"PAYMENT", "CHARGE", "TRANSFER", "REFUND", "PAYOUT"})
# Verbs that only read, when nothing else in the name writes.
_READS = frozenset(
    {"LIST", "GET", "FETCH", "SEARCH", "FIND", "READ", "QUERY", "LOOKUP", "DESCRIBE", "COUNT", "VIEW",
     "RETRIEVE", "CHECK", "DOWNLOAD", "EXPORT"}
)
WRITES = frozenset(
    {"CREATE", "UPDATE", "ADD", "MOVE", "SET", "MODIFY", "EDIT", "PATCH", "PUT", "INSERT", "UPLOAD", "WRITE",
     "RUN", "EXECUTE", "APPLY", "ARCHIVE", "CANCEL", "MARK", "LABEL", "SCHEDULE", "MERGE", "CLOSE", "RENAME",
     "COPY", "REPLACE", "SUBMIT", "APPROVE", "ASSIGN", "STAR", "ENABLE", "DISABLE", "REVOKE", "GRANT"}
)

SENDS_REASON ="Sends or shares something outside this chat"
DELETES_REASON = "Deletes data"
PAYS_REASON = "Moves money"
CHANGES_REASON = "Could change data or contact someone"
UNCHECKED_REASON = "Couldn't check what this does, so it asks"
TOO_LONG_REASON = "Too long to check, so it asks"
UNREADABLE_REASON = "Couldn't read what this runs, so it asks"
REACHES_OUT_REASON = "Runs code that could reach outside this chat"



@dataclass(frozen=True)
class Verdict:
    kind: Kind
    reason: str = ""


def words(key: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
    return [word for word in re.split(r"[^A-Za-z0-9]+", spaced.upper()) if word]


def _first(words: list[str], named: frozenset[str]) -> int | None:
    return next((index for index, word in enumerate(words) if word in named), None)


SecondCheck = Callable[[Action], Optional[Verdict]]


@dataclass(frozen=True)
class Policy:
    # The tools that change nothing, by where they come from, so another server's tool
    # of the same name gets no pass (actions.py).
    read_only: Mapping[str, frozenset[str]]
    # Where a read-only hint settles a call on its own. A person's own server writes its
    # hints like its names, so there a hint counts only for a name with no effect in it:
    # it can't mark update_record or create_invoice read only. A destructive hint asks
    # from anywhere, since it only adds a question.
    trusted_hints: frozenset[str]
    # Asked in turn for a call the classifier would rate: the first that answers decides.
    second_checks: Sequence[SecondCheck] = ()

    def verdict_of(self, action: Action, hints: frozenset[str]) -> Verdict:
        if not action.readable:
            return Verdict("ask", UNREADABLE_REASON)
        if action.key in self.read_only.get(action.source, frozenset()):
            return Verdict("run")
        named = words(own_name(action))
        read_at = _first(named, _READS)
        for verbs, reason in ((_PAYS, PAYS_REASON), (_DELETES, DELETES_REASON), (_SENDS, SENDS_REASON)):
            verb_at = _first(named, verbs)
            if verb_at is not None and (read_at is None or verb_at < read_at):
                return Verdict("ask", reason)
        if "destructiveHint" in hints:
            return Verdict("ask", DELETES_REASON)
        effects = _SENDS | _DELETES | _PAYS | _SEND_NOUNS | _PAY_NOUNS | WRITES
        if "readOnlyHint" in hints and (action.source in self.trusted_hints or not set(named) & effects):
            return Verdict("run")
        if read_at is None:
            for nouns, reason in ((_PAY_NOUNS, PAYS_REASON), (_SEND_NOUNS, SENDS_REASON)):
                if set(named) & nouns:
                    return Verdict("ask", reason)
        elif not set(named) & effects:
            # Only a name with nothing in it but reading runs on the name alone.
            return Verdict("run")
        return Verdict("classify", CHANGES_REASON)

    def second_check(self, action: Action) -> Verdict | None:
        """Asks on its own for a call the classifier would rate, where that call's text
        can steer the classifier. Its verdict from the classifier is advice, so both
        have to agree before one runs unasked.

        Another server's tool always asks: its name and description are its author's
        words. The rest are the checks this policy was built with.
        """
        if action.source == SERVER:
            return Verdict("ask", CHANGES_REASON)
        for check in self.second_checks:
            verdict = check(action)
            if verdict is not None:
                return verdict
        return None

    def decide(
        self,
        actions: list[Action],
        hints: Mapping[str, frozenset[str]],
        settings: ApprovalSettings,
    ) -> Verdict:
        """What the call as a whole does: it asks if any action asks."""
        if settings.mode == "off":
            return Verdict("run")
        if actions and all(action.readable and action.key in settings.allowed for action in actions):
            return Verdict("run")
        verdicts = [self.verdict_of(action, hints.get(action.key, frozenset())) for action in actions]
        asked = next((verdict for verdict in verdicts if verdict.kind == "ask"), None)
        if asked:
            return asked
        if all(verdict.kind == "run" for verdict in verdicts):
            return Verdict("run")
        if settings.mode == "strict":
            return Verdict("ask", CHANGES_REASON)
        return Verdict("classify", CHANGES_REASON)

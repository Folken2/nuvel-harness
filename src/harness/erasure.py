"""Removes what the agent keeps for a conversation or a person, for good.

What it holds, and where:

- ``sessions``, ``events`` and ``user_states`` in the session database: every run
  and its history, and the per-user state row ADK creates with a person's first
  session. ``app_states`` belongs to the app, not to anyone.
- A conversation's background tasks: sessions named after it (runs/tasks.py), which
  go with it. Tasks don't get the file tools, so they have no files.
- Files, in the artifact store it is given to clear. A store someone else clears is
  not given: when the files are rows in the web app's own database, they go with its
  conversation and user rows, so the server passes none (``store_to_clear``).

Memory has no store to clear: nothing writes to ADK's memory service, which is
in-process. Everything here can run twice: a second run finds nothing and removes
nothing, so a delete that failed partway through is retried by doing it again.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

from google.adk.artifacts.base_artifact_service import BaseArtifactService
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from .runs.registry import TASK_SESSION_MARK

logger = logging.getLogger(__name__)

# ADK's marker for a file that belongs to the person, not to one conversation.
_USER_SCOPED_PREFIX = "user:"


@dataclass
class Erased:
    """How much went, for the log and the caller. Zero on a repeat."""

    sessions: int = 0
    events: int = 0
    user_states: int = 0
    files: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class AgentDataEraser:
    def __init__(
        self,
        engine: Optional[AsyncEngine],
        files_to_clear: Optional[BaseArtifactService],
    ) -> None:
        # No engine: DEV_MODE's in-memory sessions, which die with the process.
        self._engine = engine
        # None: no files are kept, or their store is someone else's to clear.
        self._files = files_to_clear

    async def erase_conversation(
        self, app_name: str, user_id: str, session_id: str
    ) -> Erased:
        # Files first: if this stops partway, the session row is still there to
        # say what to retry.
        erased = Erased(files=await self._erase_files(app_name, user_id, session_id))
        await self._delete_rows(erased, app_name, user_id, session_id)
        logger.info("erased a conversation: %s", erased.as_dict())
        return erased

    async def erase_user(self, app_name: str, user_id: str) -> Erased:
        erased = Erased()
        for session_id in await self._session_ids(app_name, user_id):
            erased.files += await self._erase_files(app_name, user_id, session_id)
        erased.files += await self._erase_files(app_name, user_id, None)
        await self._delete_rows(erased, app_name, user_id, None)
        logger.info("erased a user: %s", erased.as_dict())
        return erased

    async def _erase_files(
        self, app_name: str, user_id: str, session_id: Optional[str]
    ) -> int:
        """One conversation's files, or the person's own when session_id is None."""
        if self._files is None:
            return 0
        keys = await self._files.list_artifact_keys(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        # A conversation's listing also names the person's own files, which
        # outlive it.
        if session_id is not None:
            keys = [key for key in keys if not key.startswith(_USER_SCOPED_PREFIX)]
        for key in keys:
            await self._files.delete_artifact(
                app_name=app_name, user_id=user_id, filename=key, session_id=session_id
            )
        return len(keys)

    async def _session_ids(self, app_name: str, user_id: str) -> list[str]:
        if self._engine is None:
            return []
        async with self._engine.connect() as connection:
            if not await _tables_exist(connection):
                return []
            result = await connection.execute(
                text("SELECT id FROM sessions WHERE app_name = :app AND user_id = :user"),
                {"app": app_name, "user": user_id},
            )
            return [row[0] for row in result]

    async def _delete_rows(
        self, erased: Erased, app_name: str, user_id: str, session_id: Optional[str]
    ) -> None:
        """All of it or none of it: one transaction."""
        if self._engine is None:
            logger.warning("erase with in-memory sessions: they go when the process does")
            return
        scope = {"app": app_name, "user": user_id, "session": session_id}
        # With no session id the clause is true, so the delete covers the person.
        match = "app_name = :app AND user_id = :user"
        # A conversation's background tasks (runs/tasks.py) have sessions named after it.
        tasks = f"starts_with({{column}}, CAST(:session AS text) || '{TASK_SESSION_MARK}')"
        events_match = (
            f"{match} AND (CAST(:session AS text) IS NULL OR session_id = :session"
            f" OR {tasks.format(column='session_id')})"
        )
        sessions_match = (
            f"{match} AND (CAST(:session AS text) IS NULL OR id = :session"
            f" OR {tasks.format(column='id')})"
        )
        async with self._engine.begin() as connection:
            if not await _tables_exist(connection):
                return
            result = await connection.execute(
                text(f"DELETE FROM events WHERE {events_match}"), scope
            )
            erased.events = result.rowcount
            result = await connection.execute(
                text(f"DELETE FROM sessions WHERE {sessions_match}"), scope
            )
            erased.sessions = result.rowcount
            if session_id is None:
                result = await connection.execute(
                    text(f"DELETE FROM user_states WHERE {match}"),
                    {"app": app_name, "user": user_id},
                )
                erased.user_states = result.rowcount


async def _tables_exist(connection: AsyncConnection) -> bool:
    """ADK creates its tables on first use: before that, nothing is stored."""
    result = await connection.execute(text("SELECT to_regclass('sessions')"))
    return result.scalar() is not None

"""Drives ADK's own /run_sse in-process, with this module in place of the client.

Everything about a run stays ADK's: request parsing, the session check, event
formatting. The one thing that changes is who's on the other end of the connection.
A person leaving no longer disconnects it; only a Stop does, through the same
http.disconnect a dropped client would send, so a Stop takes ADK's usual path.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

from .model_streams import closing_model_streams
from .registry import Run, RunStatus
from .steering import reset_current_inbox, set_current_inbox
from .usage_frames import reset_frame_sink, set_frame_sink

logger = logging.getLogger(__name__)

ASGIApp = Callable[..., Awaitable[None]]
RUN_PATH = "/run_sse"
# What the refusal keeps of a failed start, for the web app's log.
DETAIL_CHARS = 300


class RunStart:
    """What /run_sse answered, known before any event: the caller reports it as is."""

    def __init__(self) -> None:
        self.answered = asyncio.Event()
        self.status = 0
        self.detail = b""

    def answer(self, status: int) -> None:
        self.status = status
        self.answered.set()


async def drive_run(
    app: ASGIApp,
    run: Run,
    body: bytes,
    headers: Sequence[tuple[bytes, bytes]],
    start: RunStart,
) -> None:
    """Runs to the end whoever is watching. Never raises: the run's status says how."""
    buffered = ""
    refused = False
    body_sent = False

    async def receive() -> dict:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await run.stop_requested.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        nonlocal buffered, refused
        if message["type"] == "http.response.start":
            refused = message["status"] != 200
            if not refused:
                start.answer(message["status"])
            else:
                start.status = message["status"]
            return
        if message["type"] != "http.response.body":
            return
        chunk = message.get("body", b"")
        if refused:
            start.detail = (start.detail + chunk)[:DETAIL_CHARS]
            if not message.get("more_body", False):
                start.answered.set()
            return
        buffered += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
        while "\n\n" in buffered:
            frame, buffered = buffered.split("\n\n", 1)
            if frame:
                run.add_frame(f"{frame}\n\n")

    scope = {
        "type": "http",
        # 2.3 is what makes a streaming response listen for the disconnect above,
        # as it does under uvicorn.
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": RUN_PATH,
        "raw_path": RUN_PATH.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            *headers,
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": None,
        "server": None,
    }
    # The plugin that lets the person's messages in finds this run's inbox here.
    inbox_token = set_current_inbox(run.inbox)
    # And the ledger says here when a call's usage is in (./usage_frames.py).
    sink_token = set_frame_sink(run.add_frame)
    try:
        # ADK leaves a stopped run's model call streaming to its end: closed here.
        async with closing_model_streams():
            await app(scope, receive, send)
        run.end(RunStatus.FAILED if refused else RunStatus.FINISHED)
    except asyncio.CancelledError:
        # The service is shutting down. Not swallowed, or shutdown would wait on it.
        run.end(RunStatus.FAILED)
        raise
    except Exception:
        logger.exception("run %s failed outside ADK's own handling", run.id)
        run.end(RunStatus.FAILED)
    finally:
        reset_frame_sink(sink_token)
        reset_current_inbox(inbox_token)
        start.answered.set()

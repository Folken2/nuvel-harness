"""A model API that stops sending ends the call after the client's time limit, against a
real HTTP server on this machine that stalls (no network beyond localhost)."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litellm
import pytest

from harness.routing import retry_policy
from harness.routing.client import ClientSettings, KeyedLiteLLMClient
from harness.routing.retry_policy import Kind

LIMIT = 0.5
STALL = 5.0


def _chunk(text: str) -> bytes:
    body = {
        "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


class _Stalling(BaseHTTPRequestHandler):
    mode = "before"

    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.mode == "before":
            time.sleep(STALL)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(_chunk("Hel"))
        self.wfile.flush()
        time.sleep(STALL)


@pytest.fixture
def server():
    def serve(mode: str) -> str:
        handler = type("Handler", (_Stalling,), {"mode": mode})
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{port}/v1"

    servers: list[ThreadingHTTPServer] = []
    yield serve
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def _client() -> KeyedLiteLLMClient:
    settings = ClientSettings(
        retry=retry_policy.Settings(retries=0, first_wait=0, max_wait=0),
        retry_budget_seconds=0,
        overflow_kept_steps=1,
        request_timeout_seconds=LIMIT,
    )
    return KeyedLiteLLMClient(lambda: settings)


def _ask(base: str, **kwargs: object):
    messages = [{"role": "user", "content": "hi"}]
    return _client().acompletion("openai/m", messages, None, api_base=base, api_key="not-a-key", **kwargs)


def test_an_api_that_never_answers_times_out_as_ours(server):
    base = server("before")
    started = time.monotonic()
    with pytest.raises(litellm.Timeout) as raised:
        asyncio.run(_ask(base))
    assert time.monotonic() - started < STALL - 1
    assert retry_policy.classify(raised.value) is Kind.TIMED_OUT


def test_a_stream_that_stalls_mid_answer_ends_instead_of_hanging(server):
    base = server("mid-stream")

    async def read() -> list[str]:
        words: list[str] = []
        stream = await _ask(base, stream=True)
        async for chunk in stream:
            words.append(chunk.choices[0].delta.content or "")
        return words

    started = time.monotonic()
    with pytest.raises(Exception) as raised:
        asyncio.run(read())
    assert time.monotonic() - started < STALL - 1, raised.value

"""A POST whose answer is read only up to a size: the bytes, or TooLarge.

An endpoint that answers with more than it should (a picture far past its resolution,
a runaway completion) would otherwise be held whole in memory before anything here
could look at it. The declared length is checked first, then the bytes as they come.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class TooLarge(Exception):
    def __init__(self, limit: int, status_code: int) -> None:
        super().__init__(f"over {limit} bytes")
        self.limit = limit
        # The answer's status: a 200 was billed, whatever came after it.
        self.status_code = status_code


@dataclass(frozen=True)
class Answer:
    status_code: int
    body: bytes


async def post_capped(
    client: httpx.AsyncClient, url: str, *, headers: dict[str, str], json: Any, max_bytes: int
) -> Answer:
    async with client.stream("POST", url, headers=headers, json=json) as response:
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise TooLarge(max_bytes, response.status_code)
        chunks, size = [], 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > max_bytes:
                raise TooLarge(max_bytes, response.status_code)
            chunks.append(chunk)
        return Answer(response.status_code, b"".join(chunks))

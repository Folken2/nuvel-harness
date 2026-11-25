"""The example as an HTTP service: ADK's API server plus the harness's run routes.

    pip install uvicorn
    python examples/serve.py          # http://localhost:8000

A run started through the run routes belongs to the service, not to the request: close
the connection and it keeps going; open the stream again and it picks up where it is.
Sessions here are in memory, so nothing survives a restart. Give ``Stores`` a Postgres
URI and an interrupted run resumes after one.
"""

from __future__ import annotations

import pathlib

from fastapi import FastAPI

from harness.service import ServiceOptions, Stores, create_service

HERE = pathlib.Path(__file__).parent


def build() -> FastAPI:
    service = create_service(
        ServiceOptions(agents_dir=str(HERE), safe_to_repeat=("word_count",)),
        Stores(session_uri=None, connect_args=lambda uri: {}),
        tasks=lambda runs: None,
    )
    return service.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build(), host="127.0.0.1", port=8000)

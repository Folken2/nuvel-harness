# Auth

An agent service usually sits behind a web app that has already signed the person in. This
package checks the short-lived token that web app signs for each call, before any request
reaches the agent, and can bind a run's body to the person the token names. A product imports
`from harness.auth import InvalidToken, Refused, TokenCheck, TokenGate`.

Who may call what is the product's to say: the gate knows no route of its own.

## What it guarantees

### `TokenCheck`

- **HS256 with a shared secret, fixed.** The algorithm is never read from the token, so a
  token can't choose how it is checked.
- **Required claims.** `exp`, `iat`, `sub`, `aud` and `iss` must be present; the issuer and
  audience must be the ones the check was built with; times allow 30 seconds of clock skew.
  `sub` must be a non-empty string, and `scope` must be a string listed in `scopes`, so a check
  built with no scopes refuses every token.
- **Secret rotation.** `verify(raw, secrets)` tries each secret in order (newest first). Only a
  bad signature moves on to the next; any other problem fails at once.
- **Errors never carry the token.** Every failure is `InvalidToken` whose message is the name
  of what went wrong (such as `ExpiredSignatureError`, `no identity`, `unknown scope`).
- **The caller's own shape.** After all of that passes, `read_claims(user_id, scope, claims)`
  returns the token as the product knows it, or raises `InvalidToken` itself.
- An empty `issuer` or `audience` raises `ValueError` when the check is built, since the
  underlying JWT library would skip that check.

### `TokenGate`

Plain ASGI middleware, so it sees WebSockets too (Starlette's `BaseHTTPMiddleware` lets them
past).

- **WebSockets are closed** with code 1008, always.
- **Open paths** (exact matches in `open_paths`) and non-HTTP scopes such as lifespan pass with
  no token. So do calls the harness's background tasks make to the service itself (see
  [runs](runs.md)): they carry a mark in the ASGI scope, which no request from outside can set.
- **Every other request needs `Authorization: Bearer <token>`.** `verify` failing with
  `InvalidToken` answers 401 `{"error": "unauthorized"}`. `allows(token, method, path)`
  returning false answers 403 `{"error": "forbidden"}`.
- **Bound bodies.** For a path where `carries_body(path)` is true, the body is read whole, up to
  `max_body_bytes`. A declared `Content-Length` over the limit is refused before a byte is read,
  and a body without one is refused as soon as it passes it: 413
  `{"error": "body_too_large"}`. `bind_body(token, body)` then returns the body to pass on
  (it may rewrite it, and `Content-Length` is set to match) or raises `Refused(status, code)`,
  answered as `{"error": code}` with that status. After the body, the app reads the real
  channel, so a disconnect still reaches it.
- **Logs name no caller data.** A refusal's log line names only the method and the first path
  segment, and only when it is a plain word. Refusals without a valid token (anyone can send
  those) are capped at 20 lines a minute, and the first line after the minute says how many
  went unlogged. Refusals of a valid token are logged every time.
- The gate reads a key name from `wire_names`, so a product calls
  `wire_names.use_namespace(...)` at start-up first (see the [README](README.md)); without it
  the first request raises `RuntimeError`.

## What it costs

- One HMAC check per secret tried, per request. No network, no database.
- For a path that carries a body, the whole body is held in memory, at most `max_body_bytes`,
  before the app sees it.
- Log lines as above, capped for requests without a valid token.

## Configure it

`TokenCheck(*, issuer, audience, scopes, read_claims)` and `TokenGate(app, *, secrets,
max_body_bytes, verify, allows, carries_body, bind_body, open_paths=frozenset())`. Everything
but `open_paths` is required.

```python
import json
from dataclasses import dataclass

from harness import wire_names
from harness.auth import Refused, TokenCheck, TokenGate

wire_names.use_namespace("myproduct")


@dataclass(frozen=True)
class Caller:
    user_id: str
    scope: str


def read_claims(user_id: str, scope: str, claims) -> Caller:
    return Caller(user_id, scope)


check = TokenCheck(issuer="myproduct-web", audience="myproduct-agent",
                   scopes=frozenset({"chat"}), read_claims=read_claims)


def bind_body(caller: Caller, body: bytes) -> bytes:
    if json.loads(body).get("user_id") != caller.user_id:
        raise Refused(403, "wrong_user")
    return body


app = TokenGate(
    service_app,                          # the product's ASGI app
    secrets=[new_secret, old_secret],     # newest first
    max_body_bytes=1_000_000,
    verify=check.verify,
    allows=lambda caller, method, path: path.startswith("/run"),
    carries_body=lambda path: path == "/run",
    bind_body=bind_body,
    open_paths=frozenset({"/health"}),
)
```

## Test code that uses it

- Mint tokens in the test with PyJWT (`jwt.encode(claims, secret, algorithm="HS256")`) and a
  test secret; set `exp` in the past to test expiry.
- `TokenCheck.verify` is a plain function: call it directly and assert on the value
  `read_claims` returns or on `InvalidToken`.
- Wrap a small Starlette app in `TokenGate` and drive it with Starlette's `TestClient`; set the
  wire namespace first, as `tests/conftest.py` does. `allows`, `carries_body` and `bind_body`
  are plain callables a test can replace.
- The harness has no test file for this package yet.

## Reference

| Name | What it is |
| --- | --- |
| `TokenCheck` | One issuer's HS256 tokens for one audience, read into the caller's own type. |
| `InvalidToken` | Expired, forged, malformed or missing a claim. |
| `TokenGate` | ASGI middleware: token, permission and body checks before the app. |
| `Refused` | Raised by `bind_body`: the answer's `status` and `code`. |

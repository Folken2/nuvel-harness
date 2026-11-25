# Clients

An endpoint that answers with far more than it should (a picture past its resolution, a
runaway completion) would otherwise be held whole in memory before the code could look at it.
This package has one helper: a POST whose answer is read only up to a size. A product imports
`from harness.clients import TooLarge, post_capped`.

## What it guarantees

- `post_capped(client, url, *, headers, json, max_bytes)` sends a POST with the given headers
  and JSON body through the `httpx.AsyncClient` you pass, and returns an answer with
  `status_code` and `body` (the bytes).
- A declared `Content-Length` over `max_bytes` raises `TooLarge` before any of the body is
  read. Without one, the body is read as it streams, and `TooLarge` is raised as soon as it
  passes `max_bytes`.
- `TooLarge` carries `limit` and the answer's `status_code`: a 200 that was cut off was still
  served (and possibly billed).
- It doesn't raise for an error status: the caller reads `status_code`.

## What it costs

Nothing beyond the request: at most `max_bytes` of the body is held in memory. Retries,
timeouts and connection reuse are whatever the `httpx.AsyncClient` you pass is set up with.

## Configure it

```python
import httpx

from harness.clients import TooLarge, post_capped


async def fetch_image(client: httpx.AsyncClient, prompt: str) -> bytes | None:
    try:
        answer = await post_capped(client, "https://images.example.test/v1/generate",
                                   headers={"Authorization": f"Bearer {api_key}"},
                                   json={"prompt": prompt}, max_bytes=20 * 1024 * 1024)
    except TooLarge as error:
        log.warning("image over %d bytes (status %d)", error.limit, error.status_code)
        return None
    return answer.body if answer.status_code == 200 else None
```

## Test code that uses it

Pass an `httpx.AsyncClient(transport=httpx.MockTransport(handler))`, where `handler` returns
an `httpx.Response` of the size you want to test. No network. The harness has no test file
for this package yet.

## Reference

| Name | What it is |
| --- | --- |
| `post_capped` | A POST whose answer is read up to `max_bytes`. |
| `TooLarge` | The answer passed the limit: `limit` and `status_code`. |

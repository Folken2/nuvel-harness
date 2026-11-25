"""MCP Apps: what a server's page answer must be for the host to keep it."""

import base64

from mcp import types

from harness.tools.mcp_apps.limits import PageLimits
from harness.tools.mcp_apps.page import Page, check_page

URI = "ui://crm/pipeline"
HTML = "<!doctype html><html><body>Pipeline</body></html>"
LIMITS = PageLimits(max_page_bytes=200, max_domains=2)


def _text(text=HTML, uri=URI, mime="text/html;profile=mcp-app", meta=None):
    return types.TextResourceContents(uri=uri, mimeType=mime, text=text, _meta=meta)


def test_a_page_is_kept_with_its_hash_and_only_well_formed_origins():
    meta = {"ui": {"prefersBorder": True, "csp": {
        "connectDomains": ["https://api.crm.example", "wss://live.crm.example", "http://plain.example", "https://a.example/path"],
        "resourceDomains": ["https://*.cdn.example", "https://cdn.example", "https://third.example"],
        "frameDomains": "https://not-a-list.example",
    }}}
    page = check_page(URI, [_text(meta=meta)], LIMITS)
    assert isinstance(page, Page)
    assert page.html == HTML and page.prefers_border is True and len(page.sha256) == 64
    assert page.csp == {
        "connectDomains": ["https://api.crm.example", "wss://live.crm.example"],
        "resourceDomains": ["https://*.cdn.example", "https://cdn.example"],  # capped at two
        "frameDomains": [],
    }


def test_base64_content_and_a_mime_type_with_extra_parameters_are_read():
    blob = types.BlobResourceContents(uri=URI, mimeType="text/html; charset=utf-8; profile=mcp-app", blob=base64.b64encode(HTML.encode()).decode())
    assert isinstance(check_page(URI, [blob], LIMITS), Page)


def test_anything_else_is_refused_with_a_reason():
    cases = [
        ("https://crm.example/page", [_text()]),
        (URI, []),
        (URI, [_text(), _text()]),
        (URI, [_text(uri="ui://crm/other")]),
        (URI, [_text(mime="text/html")]),
        (URI, [_text(mime="application/javascript;profile=mcp-app")]),
        (URI, [_text(text="x" * 201)]),
        (URI, [_text(text="   ")]),
        (URI, [types.BlobResourceContents(uri=URI, mimeType="text/html;profile=mcp-app", blob="not base64!")]),
        (URI, [types.BlobResourceContents(uri=URI, mimeType="text/html;profile=mcp-app", blob=base64.b64encode(b"\xff\xfe").decode())]),
        (URI, "not a list"),
    ]
    for uri, contents in cases:
        result = check_page(uri, contents, LIMITS)
        assert isinstance(result, str), (uri, contents)

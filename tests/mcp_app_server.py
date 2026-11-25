"""An example MCP App server for tests and live checks.

One tool for the model with a page (``show_pipeline``), one for its page only
(``move_deal``), and the page: a small MCP App that does the handshake with its host,
shows the tool's result, and reports its size. Run it on its own for a live check:

    python tests/mcp_app_server.py 8765
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

PAGE_URI = "ui://pipeline/board"
PAGE_MIME = "text/html;profile=mcp-app"

DEALS = [
    {"name": "Acme renewal", "stage": "Proposal", "value": 42000},
    {"name": "Globex pilot", "stage": "Negotiation", "value": 18500},
    {"name": "Initech upsell", "stage": "Won", "value": 9900},
]

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline</title>
<style>
  body { font: 14px system-ui, sans-serif; margin: 0; padding: 12px; color: #1d1d1f; }
  body.dark { color: #f2f2f2; }
  h1 { font-size: 15px; margin: 0 0 8px; }
  table { border-collapse: collapse; width: 100%; }
  td, th { text-align: left; padding: 6px 8px; border-bottom: 1px solid #8884; }
  td.v { text-align: right; font-variant-numeric: tabular-nums; }
  #status { font-size: 12px; opacity: .7; margin-top: 8px; }
  a { color: inherit; }
</style></head>
<body>
<h1>Pipeline board</h1>
<table><thead><tr><th>Deal</th><th>Stage</th><th>Value</th></tr></thead><tbody id="rows"><tr><td colspan="3">Waiting for the host...</td></tr></tbody></table>
<p id="status">Not connected</p>
<p><a href="#" id="docs">Open the CRM docs</a> · <a href="#" id="move">Move Acme to Won</a></p>
<script>
  let next = 1;
  const pending = new Map();
  function send(method, params) {
    const id = next++;
    window.parent.postMessage({ jsonrpc: "2.0", id, method, params }, "*");
    return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
  }
  function notify(method, params) { window.parent.postMessage({ jsonrpc: "2.0", method, params }, "*"); }
  function show(result) {
    const deals = (result && result.structuredContent && result.structuredContent.deals) || [];
    document.getElementById("rows").innerHTML = deals.map(d =>
      `<tr><td>${d.name}</td><td>${d.stage}</td><td class="v">${d.value.toLocaleString()}</td></tr>`).join("");
    reportSize();
  }
  function reportSize() { notify("ui/notifications/size-changed", { width: document.body.scrollWidth, height: document.body.scrollHeight }); }
  window.addEventListener("message", (event) => {
    const m = event.data;
    if (!m || m.jsonrpc !== "2.0") return;
    if (m.id !== undefined && pending.has(m.id)) {
      const p = pending.get(m.id); pending.delete(m.id);
      m.error ? p.reject(m.error) : p.resolve(m.result);
      return;
    }
    if (m.method === "ui/notifications/tool-result") show(m.params);
    if (m.method === "ui/notifications/tool-input") document.getElementById("status").textContent = "Input: " + JSON.stringify(m.params.arguments);
    if (m.method === "ui/notifications/host-context-changed" && m.params.theme) document.body.className = m.params.theme;
  });
  document.getElementById("docs").addEventListener("click", (e) => {
    e.preventDefault();
    send("ui/open-link", { url: "https://modelcontextprotocol.io/" }).catch(err => document.getElementById("status").textContent = "Link refused: " + err.message);
  });
  document.getElementById("move").addEventListener("click", (e) => {
    e.preventDefault();
    send("tools/call", { name: "move_deal", arguments: { deal: "Acme renewal", stage: "Won" } })
      .then(() => document.getElementById("status").textContent = "Moved")
      .catch(err => document.getElementById("status").textContent = "Refused: " + err.message);
  });
  send("ui/initialize", { appInfo: { name: "Pipeline", version: "1.0.0" }, appCapabilities: {}, protocolVersion: "2026-01-26" })
    .then((result) => {
      document.getElementById("status").textContent = "Connected to " + ((result.hostInfo && result.hostInfo.name) || "host");
      if (result.hostContext && result.hostContext.theme) document.body.className = result.hostContext.theme;
      notify("ui/notifications/initialized", {});
      reportSize();
    })
    .catch((err) => document.getElementById("status").textContent = "Handshake failed: " + err.message);
</script>
</body></html>
"""


def build(port: int = 0) -> FastMCP:
    server = FastMCP("pipeline", host="127.0.0.1", port=port, stateless_http=True)

    @server.tool(
        description="Shows the sales pipeline as a board.",
        meta={"ui": {"resourceUri": PAGE_URI, "visibility": ["model", "app"]}},
    )
    def show_pipeline(region: str = "all") -> CallToolResult:
        text = "; ".join(f"{d['name']}: {d['stage']}, {d['value']}" for d in DEALS)
        return CallToolResult(content=[TextContent(type="text", text=text)], structuredContent={"region": region, "deals": DEALS})

    @server.tool(description="Moves a deal to a stage.", meta={"ui": {"visibility": ["app"]}})
    def move_deal(deal: str, stage: str) -> str:
        return f"Moved {deal} to {stage}"

    @server.resource(PAGE_URI, name="pipeline-board", mime_type=PAGE_MIME, meta={"ui": {"prefersBorder": True, "csp": {}}})
    def board() -> str:
        return PAGE

    return server


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 8765).run(transport="streamable-http")

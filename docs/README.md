# nuvel-harness docs

The harness is the production layer around an agent built on Google's Agent Development Kit
(ADK). ADK runs the agent loop; the harness adds what a product needs around it: runs that
survive a dropped connection or a restart, model routing and retry, context management, usage
metering, approval, and a set of tools and plugins.

Start with [Build your own agent](build-an-agent.md): it builds a small product, serves it, and
tests a whole run with a scripted model.

## How the pieces fit

A product is an ADK `App` plus the harness's service around it.

- **The product says who it is, once, at start-up**: its wire namespace and its file layout
  ([service.md](service.md), [files.md](files.md)). The harness reads no environment variable,
  so every other setting is passed in as an argument or a settings object.
- **The service** ([service.md](service.md)) is ADK's own API server with the harness's run
  routes, health check and erasure routes added.
- **Runs** ([runs.md](runs.md)) belong to the service rather than the HTTP request. A run keeps
  going when the browser drops, resumes from the saved session after a restart, can take a
  message mid-run, and can be stopped.
- **Plugins** ([plugins.md](plugins.md)) are where the harness meets ADK's agent loop. Each is
  an ADK `BasePlugin`. Most of the other packages (routing, context, replies, ledger, approval,
  files, tools) hold the logic, and a plugin attaches it to ADK's callbacks.
- **Tools** ([tools.md](tools.md)) are ADK tools and toolsets the agent calls: a plan, a
  question for the person, background tasks, MCP servers, skills and more.

Every package lists its public names in `__all__`. Those names are the API; anything else can
move without notice. Where the harness ships its own piece beside an ADK built-in, and why, is
in [adk.md](adk.md).

## Pages

| Page | Package | What it covers |
| --- | --- | --- |
| [build-an-agent.md](build-an-agent.md) | | A walkthrough: agent, service, test |
| [service.md](service.md) | `harness.service`, `wire_names`, `run_state`, `erasure`, `engine_limits`, `agent_without_identity` | The HTTP service, start-up names, erasure, database time limits |
| [runs.md](runs.md) | `harness.runs` | Durable runs, resume, steering, background tasks, loop guard |
| [routing.md](routing.md) | `harness.routing` | Per-turn model routes, keyed clients, retry policy |
| [context.md](context.md) | `harness.context` | Compaction, masking old tool results, prompt cache, window budget |
| [replies.md](replies.md) | `harness.replies` | Repairing replies a model got slightly wrong |
| [ledger.md](ledger.md) | `harness.ledger`, `harness.billing` | Pricing and recording every model, tool and compaction call |
| [approval.md](approval.md) | `harness.approval` | Asking the person before a tool with effects runs (changing) |
| [tools.md](tools.md) | `harness.tools` | The tools and toolsets the agent calls |
| [plugins.md](plugins.md) | `harness.plugins` | Every plugin, what it hooks and where order matters |
| [files.md](files.md) | `harness.files` | A conversation's files and their layout |
| [library.md](library.md) | `harness.library` | Reading plugin archives, repositories and marketplaces |
| [voice.md](voice.md) | `harness.voice` | Endpointing, clip transcription, sentence-by-sentence speech |
| [auth.md](auth.md) | `harness.auth` | Checking the caller's token on the service |
| [clients.md](clients.md) | `harness.clients` | HTTP calls with a cap on the response size |
| [adk.md](adk.md) | | Where the harness goes beyond ADK, and why |

Each component page has the same sections: what it guarantees, what it costs, how to
configure it, and how to test code that uses it.

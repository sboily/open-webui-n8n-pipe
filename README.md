# N8N Pipe for Open-WebUI

[![CI](https://github.com/sboily/open-webui-n8n-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/sboily/open-webui-n8n-pipe/actions/workflows/ci.yml)

An Open-WebUI [pipe function](https://docs.openwebui.com/features/plugin/functions/) that forwards chat messages — text, images and uploaded files — to an n8n workflow through a Webhook node, and returns the workflow answer to the chat.

Based on https://openwebui.com/f/coleam/n8n_pipe. Many thanks to the original author.

Hosted on the Open-WebUI hub: https://openwebui.com/f/quintana/n8n_pipe_ng

## Features

- Asynchronous `httpx` calls; the timeout valve applies to every request (no restart needed)
- Retries on connection errors and HTTP 502/503/504 only — never after a timeout, so a slow workflow is not started twice
- Heartbeat status in the chat while n8n is working, clear translated errors (English, French)
- Stable `sessionId` per chat (`__chat_id__`), isolated sessions for Open-WebUI background tasks (title, tags...)
- Images (`data:` URLs, optionally remote URLs) and uploaded documents sent as `multipart/form-data`
- Optional streaming of the answer (n8n Webhook node "Streaming response")
- Metadata (user, chat, message, task, model) forwarded to the workflow

## Installation

Pick one:

1. **Open-WebUI hub** — open https://openwebui.com/f/quintana/n8n_pipe_ng and click *Get*.
2. **Import from link** — in *Admin Panel → Functions → Import from Link*, paste `https://github.com/sboily/open-webui-n8n-pipe/releases/latest/download/n8n_pipe.py` (the bundle attached to the [latest release](https://github.com/sboily/open-webui-n8n-pipe/releases/latest)), review the code and save.
3. **From source** — `python scripts/build_single_file.py` bundles the `n8n_pipe` package into `dist/n8n_pipe.py`.

`httpx` and `pydantic` ship with Open-WebUI; nothing else is required.

## Configuration (valves)

| Valve | Description | Default |
|-------|-------------|---------|
| `n8n_host` | Base URL of the n8n server | `http://localhost:5678` |
| `n8n_webhook_id` | Path of the Webhook node (after `/webhook/`) | `your-webhook-id` |
| `n8n_test_mode` | Call `/webhook-test/` (n8n *Test workflow*) instead of `/webhook/` | `false` |
| `n8n_bearer_token` | Sent as `Authorization: Bearer ...`; empty sends no header | `""` |
| `input_field` | Field carrying the user question | `chatInput` |
| `response_field` | Field carrying the answer in the n8n JSON | `output` |
| `timeout` | Seconds to wait for the n8n answer | `120` |
| `max_retries` | Retries on connection errors / 502 / 503 / 504 | `2` |
| `emit_interval` | Seconds between "Waiting for n8n..." heartbeats | `2` |
| `enable_status_indicator` | Emit status events to the chat | `true` |
| `session_id_mode` | `chat_id` (stable per chat) or `legacy` (user id + first message) | `chat_id` |
| `stream_response` | Read the answer as a stream (see [Streaming](#streaming)) | `false` |
| `allow_remote_images` | Download `http(s)` image URLs found in messages | `false` |
| `max_attachment_size_mb` | Maximum size of one attachment | `10` |
| `language` | Language of chat messages: `en`, `fr` | `en` |

The webhook URL is `{n8n_host}/webhook/{n8n_webhook_id}` (or `/webhook-test/` in test mode).

## What n8n receives

Without attachments, a JSON body:

```json
{
  "sessionId": "b2c1e9f0-...",
  "chatInput": "What time is it?",
  "metadata": {
    "user_id": "a1b2c3",
    "chat_id": "b2c1e9f0-...",
    "message_id": "d4e5f6",
    "task": null,
    "model": "n8n_pipe_ng"
  }
}
```

With attachments, the same fields as `multipart/form-data` (`metadata` is a JSON string) plus one file part per attachment:

| Part | Content |
|------|---------|
| `image_0`, `image_1`, ... | Images attached to the last message (`image_0.png`, `image_1.jpg`, ...) |
| `file_0`, `file_1`, ... | Documents uploaded in the chat, with their original filename and MIME type |

The Webhook node parses `multipart/form-data` by itself: fields are in `$json.body`, files in the item's binary data under the part name (`image_0`, `file_0`, ...).

### Session id

- `chat_id` mode (default): the Open-WebUI chat id — stable for the whole conversation and unique across chats. Use it as the memory key of your agent.
- `legacy` mode: `"{user_id} - {first message truncated to 100 chars}"`, the format used by versions ≤ 0.3. Two chats starting with the same sentence share a session.
- Background tasks run by Open-WebUI on the same model (title, tags, follow-ups...) get a separate session, `"{sessionId}:{task}"`, and `metadata.task` is set (`title_generation`, `tags_generation`, ...). Configure a dedicated *task model* in Open-WebUI (*Admin → Settings → Interface*) if you do not want these calls to reach n8n at all.

## What n8n must return

A JSON object containing the `response_field`:

```json
{ "output": "Text answer" }
```

- Webhook node *Respond: Using 'Respond to Webhook' node* or *When Last Node Finishes / First Entry JSON*. A list (`[{"output": ...}]`, *All Entries*) is accepted: the first item is used.
- A non-string value is serialised as JSON before being shown.

### Streaming

Set the Webhook node to *Respond: Streaming response*, enable streaming on the AI Agent node, and turn on the `stream_response` valve. The pipe reads n8n's newline-delimited JSON (`{"type": "item", "content": "..."}`) and shows tokens as they arrive; `{"type": "error"}` chunks are reported to the chat. Plain-text streams and classic JSON answers keep working with the valve enabled.

## Status messages and errors

Status events (`{"type": "status", "data": {"level", "description", "done"}}`) show progress, retries, heartbeats and the final state. Errors are raised with a user-facing message, which Open-WebUI displays in the chat; the Open-WebUI log (`n8n_pipe.*` loggers) has the details.

## Attachments and security

- Only `image/*` content is accepted for images; every attachment is limited to `max_attachment_size_mb`.
- Remote image URLs are disabled by default. When enabled, hosts resolving to private, loopback or link-local addresses are refused.
- Uploaded documents are read through Open-WebUI's storage. When the bytes are not available, the text Open-WebUI extracted from the document is sent as `{name}.txt`.

## Example workflow

[`examples/n8n_echo_workflow.json`](examples/n8n_echo_workflow.json) is a minimal importable workflow: a Webhook node (POST, header auth) answering `{"output": "Echo: ..."}`. Import it in n8n, create a *Header Auth* credential (`Authorization` / `Bearer <token>`), set the same token in `n8n_bearer_token`, and set `n8n_webhook_id` to `openwebui`.

## Troubleshooting

| Message | Cause / fix |
|---------|-------------|
| `n8n did not answer within 120.0s` | Raise `timeout`. Also check the timeouts of any reverse proxy in front of Open-WebUI or n8n. |
| `n8n is unavailable after N attempts` | n8n unreachable or returning 502/503/504; check the host and that the workflow is active. |
| `n8n returned HTTP 404` | Wrong `n8n_webhook_id`, or the workflow is not active (production URLs need an active workflow; use `n8n_test_mode` while testing). |
| `n8n returned HTTP 401/403` | Bearer token mismatch with the Webhook node credential. |
| `Response field 'output' not found` | The workflow answer has no `output` key; adjust `response_field` or the Respond node. |
| `Remote image URLs are disabled` | Enable `allow_remote_images` if you trust the users of this model. |
| Empty title or odd chat titles | Open-WebUI runs title generation through the pipe; set a task model in Open-WebUI settings. |

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest                      # unit tests with coverage (threshold 90 %)
pre-commit run --all-files  # ruff format + lint, mypy
python scripts/build_single_file.py   # dist/n8n_pipe.py
```

Layout: `n8n_pipe/` (package: `valves`, `request`, `attachments`, `client`, `pipe`, `messages`, `status`, `constants`, `errors`), `tests/`, `scripts/build_single_file.py` (bundler), `examples/`.

Releasing: bump `__version__` in `n8n_pipe/__init__.py`, update `CHANGELOG.md`, tag the commit with the same version (`git tag 0.4.0 && git push --tags`). The release workflow checks the tag against `__version__` and attaches `n8n_pipe.py` to the GitHub release.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests are welcome.

## License

MIT — see [LICENSE](LICENSE).

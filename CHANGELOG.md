# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] - unreleased

### Breaking changes

- `sessionId` defaults to the Open-WebUI chat id (`session_id_mode: chat_id`). Set the valve to `legacy` to keep the `"{user_id} - {first message}"` format.
- Background tasks (title, tags, follow-ups) use a separate session `"{sessionId}:{task}"`.
- Errors are raised instead of returned as `{"error": ...}`; Open-WebUI shows the message in the chat.
- The `"Prompt: "` prefix is no longer stripped from messages.
- `n8n_bearer_token` defaults to empty and the `Authorization` header is omitted when empty.
- `history_limit`, the `body["messages"]` mutation and the async context manager were removed (they had no effect in Open-WebUI).
- Default `timeout` raised from 30 s to 120 s.
- `pydantic>=2` is required (the code already used the v2 API).

### Fixed

- The `timeout` valve applies to every request; changing it no longer requires restarting Open-WebUI (#7).
- Timeouts, 4xx responses and unusable answers are not retried, so a slow workflow is not started several times (#7).
- Error messages always carry the exception type, e.g. `ReadTimeout` no longer produces an empty message (#2).
- Unexpected errors are logged with their traceback.
- n8n answers wrapped in a list (`When Last Node Finishes / All Entries`) are accepted.
- HTTP 2xx statuses other than 200 are accepted.

### Added

- Uploaded documents (`__files__`) are forwarded as multipart parts (#3).
- `metadata` object (`user_id`, `chat_id`, `message_id`, `task`, `model`) in the payload.
- Streaming of the n8n answer (`stream_response` valve, n8n newline-delimited JSON format).
- Heartbeat status while waiting for n8n; exponential backoff between retries.
- French messages (`language` valve).
- `examples/n8n_echo_workflow.json`, issue templates, CONTRIBUTING guide.

### Security

- Remote image downloads are opt-in (`allow_remote_images`), refuse private network addresses and are size-limited.
- Every attachment is limited to `max_attachment_size_mb`; images must be `image/*`; base64 is validated.

### Internal

- Source split into a package (`n8n_pipe/`), bundled into a single file by `scripts/build_single_file.py`; releases attach the bundle.
- Single version source (`n8n_pipe.__version__`), checked against the git tag by the release workflow.
- ruff replaces black/isort/flake8; CI runs pre-commit, tests on Python 3.10–3.13 with coverage, and builds the bundle; Dependabot enabled.
- Tests rewritten on `httpx.MockTransport`.

## [0.3] - 2025-06-09

- Image attachments sent as multipart (#6).
- Pydantic v2 validators.

## [0.2] - 2025-03-13

- Initial fork with async httpx client, retries and status events.

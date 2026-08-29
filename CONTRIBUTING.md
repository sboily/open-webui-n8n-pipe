# Contributing

Thanks for helping improve the n8n pipe.

## Reporting a problem

Use the *Bug report* issue template. The versions of the pipe, Open-WebUI and n8n, the Webhook node *Respond* setting and the status message shown in the chat are what make a report actionable.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

- `pytest` runs the unit tests with coverage (threshold 90 %).
- `pre-commit run --all-files` runs ruff (format + lint) and mypy; CI runs the same command.
- `python scripts/build_single_file.py` produces `dist/n8n_pipe.py`, the file installed in Open-WebUI. Never edit `dist/` by hand.

## Pull requests

- One topic per pull request, with tests for the behaviour you change (`tests/` mirrors the package modules).
- User-facing text goes through `n8n_pipe/messages.py`; add every key to every language.
- Anything the workflow receives (`sessionId`, fields, multipart parts, `metadata`) is a public contract: document changes in `README.md` and `CHANGELOG.md`.
- Keep the package bundle-friendly: modules only use relative imports of each other, and new modules are added to `MODULE_ORDER` in `scripts/build_single_file.py` (a test checks this).

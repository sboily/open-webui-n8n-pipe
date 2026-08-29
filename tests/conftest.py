"""Shared fixtures: a configured pipe, an event recorder and httpx mock transports."""

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from n8n_pipe import Pipe

Handler = Callable[[httpx.Request], httpx.Response]

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


class EventLog:
    """Records the status events emitted by the pipe."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    @property
    def descriptions(self) -> list[str]:
        return [event["data"]["description"] for event in self.events]

    @property
    def last(self) -> dict[str, Any]:
        return self.events[-1]["data"]

    def with_level(self, level: str) -> list[dict[str, Any]]:
        return [event["data"] for event in self.events if event["data"]["level"] == level]


class Recorder:
    """Mock transport handler that records requests and delegates to ``responder``."""

    def __init__(self, responder: Handler | list[httpx.Response | Exception]) -> None:
        self.requests: list[httpx.Request] = []
        self._responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if callable(self._responder):
            return self._responder(request)
        outcome = self._responder.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def last_json(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def json_response(payload: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


@pytest.fixture
def pipe() -> Pipe:
    pipe = Pipe()
    pipe.valves = Pipe.Valves(
        n8n_host="http://n8n.test:5678",
        n8n_webhook_id="hook",
        max_retries=1,
        emit_interval=0.05,
    )
    return pipe


@pytest.fixture
def events() -> EventLog:
    return EventLog()


@pytest.fixture
def body() -> dict[str, Any]:
    return {
        "model": "n8n_pipe",
        "stream": True,
        "messages": [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What time is it?"},
        ],
    }


def install_transport(pipe: Pipe, handler: Callable[..., Any]) -> None:
    pipe._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


def echo_recorder(output: Any = "answer") -> Recorder:
    return Recorder(lambda request: json_response({"output": output}))

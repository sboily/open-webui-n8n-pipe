import asyncio
import json

import httpx
import pytest

from n8n_pipe import client as client_module
from n8n_pipe.client import N8nClient, Payload
from n8n_pipe.errors import N8nPipeError
from n8n_pipe.status import StatusEmitter
from n8n_pipe.valves import Valves
from tests.conftest import EventLog, Recorder, json_response

URL = "http://n8n.test/webhook/hook"
PAYLOAD = Payload(fields={"sessionId": "s", "chatInput": "hi", "metadata": {"a": 1}})


def make_client(handler, events=None, **valves) -> tuple[N8nClient, EventLog]:
    events = events or EventLog()
    valves_model = Valves(max_retries=1, emit_interval=0.05, **valves)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return N8nClient(valves_model, http_client, StatusEmitter(events, valves_model), URL), events


@pytest.fixture
def no_backoff(monkeypatch):
    async def instant(_delay):
        return None

    monkeypatch.setattr(client_module.asyncio, "sleep", instant)


def test_payload_is_json_without_files_and_multipart_with_files():
    assert PAYLOAD.request_kwargs() == {"json": PAYLOAD.fields}
    multipart = Payload(fields=PAYLOAD.fields, files=[("image_0", ("a.png", b"x", "image/png"))])
    kwargs = multipart.request_kwargs()
    assert kwargs["data"] == {"sessionId": "s", "chatInput": "hi", "metadata": '{"a": 1}'}
    assert kwargs["files"] == multipart.files


async def test_send_returns_the_configured_response_field():
    recorder = Recorder(lambda request: json_response({"output": "42"}))
    n8n, _ = make_client(recorder)
    assert await n8n.send(PAYLOAD) == "42"
    assert recorder.last_json == PAYLOAD.fields
    assert recorder.requests[0].headers["content-type"] == "application/json"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ([{"output": "from list"}], "from list"),
        ({"output": {"nested": True}}, '{"nested": true}'),
        ({"output": ["a", "é"]}, '["a", "é"]'),
    ],
)
async def test_send_unwraps_lists_and_serializes_non_strings(response, expected):
    n8n, _ = make_client(lambda request: json_response(response))
    assert await n8n.send(PAYLOAD) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (json_response({"wrong": "field"}), "Response field 'output' not found"),
        (json_response([]), "Response field 'output' not found"),
        (json_response("text"), "Response field 'output' not found"),
        (httpx.Response(200, content=b"<html>"), "not valid JSON"),
        (httpx.Response(401, content=b"Unauthorized"), "HTTP 401: Unauthorized"),
        (httpx.Response(404, content=b"x" * 500), "HTTP 404"),
    ],
)
async def test_send_reports_unusable_answers_without_retrying(response, expected):
    recorder = Recorder([response])
    n8n, _ = make_client(recorder)
    with pytest.raises(N8nPipeError, match=expected):
        await n8n.send(PAYLOAD)
    assert len(recorder.requests) == 1


async def test_error_body_preview_is_truncated():
    n8n, _ = make_client(lambda request: httpx.Response(500, content=b"x" * 500))
    with pytest.raises(N8nPipeError) as info:
        await n8n.send(PAYLOAD)
    assert len(str(info.value)) < 300


@pytest.mark.parametrize(
    "first_outcome",
    [
        httpx.Response(503, content=b"busy"),
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("closed"),
    ],
)
@pytest.mark.usefixtures("no_backoff")
async def test_transient_failures_are_retried_with_a_warning(first_outcome):
    recorder = Recorder([first_outcome, json_response({"output": "ok"})])
    n8n, events = make_client(recorder)
    assert await n8n.send(PAYLOAD) == "ok"
    assert len(recorder.requests) == 2
    assert events.with_level("warning")[0]["description"] == "Retrying in 1.0s (1/2)"
    assert "Attempt 2/2" in events.descriptions


@pytest.mark.usefixtures("no_backoff")
async def test_retries_are_exhausted_with_a_clear_message():
    recorder = Recorder([httpx.Response(502), httpx.Response(504)])
    n8n, _ = make_client(recorder)
    with pytest.raises(N8nPipeError, match="unavailable after 2 attempts: HTTP 504"):
        await n8n.send(PAYLOAD)
    assert len(recorder.requests) == 2


async def test_timeout_is_never_retried_and_has_a_readable_message():
    """Issue #2 (empty error) and #7 (workflow re-triggered after a timeout)."""
    recorder = Recorder([httpx.ReadTimeout(""), json_response({"output": "late"})])
    n8n, events = make_client(recorder, timeout=45)
    with pytest.raises(N8nPipeError, match=r"did not answer within 45\.0s"):
        await n8n.send(PAYLOAD)
    assert len(recorder.requests) == 1
    assert events.with_level("warning") == []


async def test_other_transport_errors_are_reported():
    n8n, _ = make_client(Recorder([httpx.TooManyRedirects("loop")]))
    with pytest.raises(N8nPipeError, match="Could not reach n8n: TooManyRedirects: loop"):
        await n8n.send(PAYLOAD)


async def test_timeout_valve_applies_to_every_request():
    """Issue #7: changing the valve must not require a restart."""
    seen: list[float] = []

    def handler(request):
        seen.append(request.extensions["timeout"]["read"])
        return json_response({"output": "ok"})

    n8n, _ = make_client(handler, timeout=30)
    await n8n.send(PAYLOAD)
    n8n._valves.timeout = 900
    await n8n.send(PAYLOAD)
    assert seen == [30, 900]


async def test_bearer_header_is_only_sent_when_configured():
    recorder = Recorder(lambda request: json_response({"output": "ok"}))
    n8n, _ = make_client(recorder)
    await n8n.send(PAYLOAD)
    assert "authorization" not in recorder.requests[0].headers
    n8n._valves.n8n_bearer_token = "s3cret"
    await n8n.send(PAYLOAD)
    assert recorder.requests[1].headers["authorization"] == "Bearer s3cret"


async def test_heartbeat_is_emitted_while_waiting():
    async def slow(request):
        await asyncio.sleep(0.2)
        return json_response({"output": "slow"})

    n8n, events = make_client(slow)
    assert await n8n.send(PAYLOAD) == "slow"
    waiting = [text for text in events.descriptions if text.startswith("Waiting for n8n")]
    assert waiting, events.descriptions


async def test_heartbeat_cancels_the_request_when_cancelled():
    started = asyncio.Event()

    async def hanging(request):
        started.set()
        await asyncio.sleep(10)
        return json_response({"output": "never"})

    n8n, _ = make_client(hanging)
    task = asyncio.ensure_future(n8n.send(PAYLOAD))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def ndjson(*chunks) -> bytes:
    return "\n".join(chunks).encode()


async def collect(iterator) -> list[str]:
    return [chunk async for chunk in iterator]


async def test_stream_yields_n8n_item_chunks():
    body = ndjson(
        json.dumps({"type": "begin", "metadata": {"nodeId": "1"}}),
        json.dumps({"type": "item", "content": "Hel"}),
        "",
        json.dumps({"type": "item", "content": "lo"}),
        json.dumps({"type": "unknown"}),
        json.dumps({"type": "end"}),
    )
    n8n, events = make_client(
        lambda request: httpx.Response(200, content=body, headers={"content-type": "text/plain"})
    )
    assert await collect(n8n.stream(PAYLOAD)) == ["Hel", "lo"]
    assert "Streaming response from n8n..." in events.descriptions


async def test_stream_passes_plain_text_and_non_object_json_lines_through():
    body = ndjson("plain text", "[1, 2]", "more")
    n8n, _ = make_client(
        lambda request: httpx.Response(200, content=body, headers={"content-type": "text/plain"})
    )
    assert await collect(n8n.stream(PAYLOAD)) == ["plain text", "[1, 2]", "more"]


async def test_stream_falls_back_to_json_answer_when_workflow_does_not_stream():
    n8n, _ = make_client(lambda request: json_response([{"output": "classic"}]))
    assert await collect(n8n.stream(PAYLOAD)) == ["classic"]


async def test_stream_error_chunk_raises():
    body = ndjson(
        json.dumps({"type": "item", "content": "partial"}),
        json.dumps({"type": "error", "content": "boom"}),
    )
    n8n, _ = make_client(
        lambda request: httpx.Response(200, content=body, headers={"content-type": "text/plain"})
    )
    with pytest.raises(N8nPipeError, match="stream reported an error: boom"):
        await collect(n8n.stream(PAYLOAD))


@pytest.mark.usefixtures("no_backoff")
async def test_stream_retries_transient_status_and_reports_http_errors():
    recorder = Recorder(
        [
            httpx.Response(503),
            httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"}),
        ]
    )
    n8n, _ = make_client(recorder)
    assert await collect(n8n.stream(PAYLOAD)) == ["ok"]
    assert len(recorder.requests) == 2

    n8n, _ = make_client(lambda request: httpx.Response(403, content=b"forbidden"))
    with pytest.raises(N8nPipeError, match="HTTP 403: forbidden"):
        await collect(n8n.stream(PAYLOAD))

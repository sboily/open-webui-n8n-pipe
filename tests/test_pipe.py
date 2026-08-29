import logging

import httpx
import pytest

from n8n_pipe import Pipe
from n8n_pipe.errors import N8nPipeError
from tests.conftest import PNG_DATA_URL, Recorder, echo_recorder, install_transport, json_response


def test_webhook_url_for_production_and_test_mode(pipe):
    assert pipe.get_webhook_url() == "http://n8n.test:5678/webhook/hook"
    pipe.valves.n8n_test_mode = True
    assert pipe.get_webhook_url() == "http://n8n.test:5678/webhook-test/hook"


def test_valves_class_is_exposed_for_open_webui():
    assert Pipe.Valves is Pipe().valves.__class__


async def test_successful_call_sends_json_and_reports_progress(pipe, body, events):
    recorder = echo_recorder("It is noon")
    install_transport(pipe, recorder)

    result = await pipe.pipe(body, {"id": "u1"}, events, "chat-1", "msg-9")

    assert result == "It is noon"
    assert recorder.requests[0].url == "http://n8n.test:5678/webhook/hook"
    assert recorder.last_json == {
        "sessionId": "chat-1",
        "chatInput": "What time is it?",
        "metadata": {
            "user_id": "u1",
            "chat_id": "chat-1",
            "message_id": "msg-9",
            "task": None,
            "model": "n8n_pipe",
        },
    }
    assert events.descriptions[:2] == ["Calling n8n workflow...", "Attempt 1/2"]
    assert events.last == {"level": "info", "description": "Complete", "done": True}


async def test_legacy_session_id_combines_user_and_first_message(pipe, body):
    pipe.valves.session_id_mode = "legacy"
    body["messages"][0]["content"] = "A" * 150
    recorder = echo_recorder()
    install_transport(pipe, recorder)
    await pipe.pipe(body, {"id": "u1"}, None, "chat-1")
    assert recorder.last_json["sessionId"] == "u1 - " + "A" * 100

    await pipe.pipe({"messages": [{"role": "user", "content": "x"}]}, None, None, None)
    assert recorder.last_json["sessionId"] == "anonymous - x"


async def test_chat_id_mode_without_chat_id_uses_legacy_formula(pipe, body):
    recorder = echo_recorder()
    install_transport(pipe, recorder)
    await pipe.pipe(body, {"id": "u1"})
    assert recorder.last_json["sessionId"] == "u1 - Hello there"


async def test_background_tasks_get_an_isolated_session(pipe, body):
    recorder = echo_recorder("Title")
    install_transport(pipe, recorder)
    await pipe.pipe(body, {"id": "u1"}, None, "chat-1", None, "title_generation")
    assert recorder.last_json["sessionId"] == "chat-1:title_generation"
    assert recorder.last_json["metadata"]["task"] == "title_generation"


async def test_images_are_sent_as_multipart(pipe, events):
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    {"type": "text", "text": "this"},
                ],
            }
        ]
    }
    recorder = echo_recorder("A picture")
    install_transport(pipe, recorder)

    assert await pipe.pipe(body, {"id": "u1"}, events, "chat-1") == "A picture"

    request = recorder.requests[0]
    assert request.headers["content-type"].startswith("multipart/form-data")
    content = request.content.decode("latin-1")
    assert 'name="sessionId"\r\n\r\nchat-1' in content
    assert 'name="chatInput"\r\n\r\nDescribe this' in content
    assert 'name="metadata"\r\n\r\n{"user_id": "u1"' in content
    assert 'name="image_0"; filename="image_0.png"\r\nContent-Type: image/png' in content


async def test_uploaded_files_are_forwarded(pipe, body):
    files = [
        {
            "type": "file",
            "id": "f1",
            "name": "notes.md",
            "file": {"id": "f1", "filename": "notes.md", "data": {"content": "hello"}},
        }
    ]
    recorder = echo_recorder()
    install_transport(pipe, recorder)
    await pipe.pipe(body, {"id": "u1"}, None, "chat-1", None, None, files)
    content = recorder.requests[0].content.decode()
    assert 'name="file_0"; filename="notes.txt"\r\nContent-Type: text/plain\r\n\r\nhello' in content


async def test_invalid_requests_raise_and_emit_an_error_status(pipe, events):
    with pytest.raises(N8nPipeError, match="must be from a user"):
        await pipe.pipe({"messages": [{"role": "assistant", "content": "x"}]}, None, events)
    assert events.last == {
        "level": "error",
        "description": "The last message must be from a user",
        "done": True,
    }


async def test_n8n_failures_raise_and_emit_an_error_status(pipe, body, events, caplog):
    install_transport(pipe, lambda request: httpx.Response(500, content=b"broken"))
    with (
        caplog.at_level(logging.WARNING, logger="n8n_pipe.pipe"),
        pytest.raises(N8nPipeError, match="HTTP 500"),
    ):
        await pipe.pipe(body, None, events)
    assert events.last["level"] == "error"
    assert "n8n call failed" in caplog.text


async def test_unexpected_errors_are_logged_with_traceback_and_described(
    pipe, body, events, caplog
):
    def explode(request):
        raise RuntimeError("boom")

    install_transport(pipe, explode)
    with caplog.at_level(logging.ERROR, logger="n8n_pipe.pipe"), pytest.raises(RuntimeError):
        await pipe.pipe(body, None, events)
    assert events.last["description"] == "Error while calling n8n: RuntimeError: boom"
    assert "Traceback" in caplog.text


async def test_french_messages(pipe, body, events):
    pipe.valves.language = "fr"
    install_transport(pipe, echo_recorder())
    await pipe.pipe(body, None, events)
    assert events.descriptions[0] == "Appel du workflow n8n..."
    assert events.last["description"] == "Terminé"


async def test_stream_mode_returns_an_async_iterator(pipe, body, events):
    pipe.valves.stream_response = True
    ndjson = b'{"type":"item","content":"Hel"}\n{"type":"item","content":"lo"}\n'
    install_transport(
        pipe,
        lambda request: httpx.Response(200, content=ndjson, headers={"content-type": "text/plain"}),
    )

    result = await pipe.pipe(body, None, events, "chat-1")

    assert not isinstance(result, str)
    assert [chunk async for chunk in result] == ["Hel", "lo"]
    assert events.last == {"level": "info", "description": "Complete", "done": True}


async def test_stream_mode_is_ignored_when_open_webui_does_not_stream(pipe, body):
    pipe.valves.stream_response = True
    body["stream"] = False
    install_transport(pipe, echo_recorder("plain"))
    assert await pipe.pipe(body, None, None, "chat-1") == "plain"


async def test_stream_failures_emit_an_error_status(pipe, body, events):
    pipe.valves.stream_response = True
    ndjson = b'{"type":"error","content":"agent crashed"}\n'
    install_transport(
        pipe,
        lambda request: httpx.Response(200, content=ndjson, headers={"content-type": "text/plain"}),
    )
    result = await pipe.pipe(body, None, events, "chat-1")
    with pytest.raises(N8nPipeError, match="agent crashed"):
        [chunk async for chunk in result]
    assert events.last["level"] == "error"


async def test_http_client_is_recreated_when_closed(pipe):
    first = pipe._get_http_client()
    assert pipe._get_http_client() is first
    await first.aclose()
    assert pipe._get_http_client() is not first
    await pipe._get_http_client().aclose()


async def test_retry_status_is_visible_end_to_end(pipe, body, events, monkeypatch):
    from n8n_pipe import client as client_module

    async def instant(_delay):
        return None

    monkeypatch.setattr(client_module.asyncio, "sleep", instant)
    install_transport(
        pipe, Recorder([httpx.Response(503), json_response({"output": "second try"})])
    )
    assert await pipe.pipe(body, None, events, "chat-1") == "second try"
    assert events.with_level("warning")[0]["description"] == "Retrying in 1.0s (1/2)"

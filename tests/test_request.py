import pytest

from n8n_pipe.constants import Language
from n8n_pipe.errors import N8nPipeError
from n8n_pipe.request import ChatRequest, image_urls_of, text_of

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_text_of_joins_text_parts_only():
    assert text_of("plain") == "plain"
    parts = [{"type": "text", "text": "Analyze"}, IMAGE, {"type": "text", "text": "this"}]
    assert text_of(parts) == "Analyze this"
    assert text_of([]) == ""


def test_image_urls_of_ignores_text_and_malformed_parts():
    assert image_urls_of("plain") == []
    parts = [{"type": "text", "text": "x"}, IMAGE, {"type": "image_url", "image_url": {}}]
    assert image_urls_of(parts) == ["data:image/png;base64,AAAA"]


def test_from_body_extracts_question_images_and_first_message():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": [{"type": "text", "text": "Look"}, IMAGE]},
        ]
    }
    request = ChatRequest.from_body(body, Language.EN)
    assert request.question == "Look"
    assert request.first_message_text == "First"
    assert request.image_urls == ["data:image/png;base64,AAAA"]


def test_from_body_accepts_image_only_message():
    body = {"messages": [{"role": "user", "content": [IMAGE]}]}
    assert ChatRequest.from_body(body, Language.EN).question == ""


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({}, "No messages found"),
        ({"messages": []}, "No messages found"),
        ({"messages": [{"role": "assistant", "content": "Hi"}]}, "must be from a user"),
        ({"messages": [{"role": "user", "content": "   "}]}, "non-empty question"),
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": " "}]}]}, "non-empty"),
        ({"messages": [{"role": "user"}]}, "non-empty question"),
    ],
)
def test_from_body_rejects_invalid_bodies(body, expected):
    with pytest.raises(N8nPipeError, match=expected):
        ChatRequest.from_body(body, Language.EN)


def test_from_body_errors_are_translated():
    with pytest.raises(N8nPipeError, match="Aucun message"):
        ChatRequest.from_body({"messages": []}, Language.FR)

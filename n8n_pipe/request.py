"""Validation and parsing of the Open-WebUI request body."""

from dataclasses import dataclass, field
from typing import Any

from .constants import ContentPartType, Language, Role
from .errors import N8nPipeError
from .messages import t

MessageContent = str | list[dict[str, Any]]


def text_of(content: MessageContent) -> str:
    """Return the text of a message content (plain string or multi-modal parts)."""
    if isinstance(content, str):
        return content
    texts = [part["text"] for part in content if part.get("type") == ContentPartType.TEXT]
    return " ".join(text for text in texts if text)


def image_urls_of(content: MessageContent) -> list[str]:
    """Return the image URLs (data: or http(s)) attached to a message content."""
    if isinstance(content, str):
        return []
    images = [part for part in content if part.get("type") == ContentPartType.IMAGE_URL]
    return [part["image_url"]["url"] for part in images if part.get("image_url", {}).get("url")]


@dataclass(frozen=True)
class ChatRequest:
    """The parts of the Open-WebUI request the pipe forwards to n8n."""

    question: str
    first_message_text: str
    image_urls: list[str] = field(default_factory=list)

    @classmethod
    def from_body(cls, body: dict[str, Any], language: Language) -> "ChatRequest":
        """Validate ``body`` and extract the question, images and first message.

        Raises:
            N8nPipeError: when the body has no messages, the last message is not
                from the user, or the question is empty without any image.
        """
        messages = body.get("messages") or []
        if not messages:
            raise N8nPipeError(t(language, "error.no_messages"))
        last = messages[-1]
        if last.get("role") != Role.USER:
            raise N8nPipeError(t(language, "error.last_not_user"))
        content = last.get("content") or ""
        question = text_of(content)
        image_urls = image_urls_of(content)
        if not question.strip() and not image_urls:
            raise N8nPipeError(t(language, "error.empty_question"))
        first_text = text_of(messages[0].get("content") or "")
        return cls(question=question, first_message_text=first_text, image_urls=image_urls)

"""Open-WebUI entry point: validates the request and orchestrates the n8n call."""

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from .attachments import AttachmentCollector
from .client import N8nClient, Payload
from .constants import (
    ANONYMOUS_USER_ID,
    LEGACY_SESSION_PREFIX_LENGTH,
    METADATA_FIELD,
    SESSION_FIELD,
    WEBHOOK_PATH,
    WEBHOOK_TEST_PATH,
    SessionIdMode,
)
from .errors import N8nPipeError
from .messages import describe, t
from .request import ChatRequest
from .status import EventEmitter, StatusEmitter
from .valves import Valves

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallContext:
    """Identifiers Open-WebUI injects for one chat completion."""

    user_id: str
    chat_id: str | None
    message_id: str | None
    task: str | None
    model: str | None

    def as_metadata(self) -> dict[str, str | None]:
        """Return the ``metadata`` object forwarded to n8n."""
        return {
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "task": self.task,
            "model": self.model,
        }


SessionStrategy = Callable[[ChatRequest, CallContext], str]


class Pipe:
    """n8n pipe connector for Open-WebUI."""

    Valves = Valves

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._http_client: httpx.AsyncClient | None = None
        self._session_strategies: dict[SessionIdMode, SessionStrategy] = {
            SessionIdMode.CHAT_ID: self._chat_session,
            SessionIdMode.LEGACY: self._legacy_session,
        }

    def get_webhook_url(self) -> str:
        """Build the webhook URL for the production or test endpoint."""
        path = WEBHOOK_TEST_PATH if self.valves.n8n_test_mode else WEBHOOK_PATH
        return f"{self.valves.n8n_host}/{path}/{self.valves.n8n_webhook_id}"

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __event_emitter__: EventEmitter | None = None,
        __chat_id__: str | None = None,
        __message_id__: str | None = None,
        __task__: str | None = None,
        __files__: list[dict[str, Any]] | None = None,
    ) -> str | AsyncIterator[str]:
        """Forward the last user message to n8n and return the workflow answer.

        Raises:
            N8nPipeError: with a user-facing message when the request is invalid
                or n8n cannot be reached; Open-WebUI displays it in the chat.
        """
        status = StatusEmitter(__event_emitter__, self.valves)
        context = CallContext(
            user_id=(__user__ or {}).get("id") or ANONYMOUS_USER_ID,
            chat_id=__chat_id__,
            message_id=__message_id__,
            task=__task__,
            model=body.get("model"),
        )
        try:
            request = ChatRequest.from_body(body, self.valves.language)
            await status.info("status.calling")
            payload = await self._build_payload(request, context, __files__ or [])
            client = N8nClient(self.valves, self._get_http_client(), status, self.get_webhook_url())
            if self.valves.stream_response and body.get("stream"):
                return self._stream(client, payload, status)
            output = await client.send(payload)
        except Exception as error:
            await self._report(status, error)
            raise
        await status.done("status.complete")
        return output

    async def _stream(
        self, client: N8nClient, payload: Payload, status: StatusEmitter
    ) -> AsyncIterator[str]:
        try:
            async for chunk in client.stream(payload):
                yield chunk
        except Exception as error:
            await self._report(status, error)
            raise
        await status.done("status.complete")

    async def _report(self, status: StatusEmitter, error: Exception) -> None:
        if isinstance(error, N8nPipeError):
            logger.warning("n8n call failed: %s", error)
            await status.error(str(error))
            return
        logger.exception("Unexpected error while calling n8n")
        await status.error(t(self.valves.language, "status.error", error=describe(error)))

    async def _build_payload(
        self, request: ChatRequest, context: CallContext, files: list[dict[str, Any]]
    ) -> Payload:
        fields: dict[str, Any] = {
            SESSION_FIELD: self._session_id(request, context),
            self.valves.input_field: request.question,
            METADATA_FIELD: context.as_metadata(),
        }
        collector = AttachmentCollector(self.valves, self._get_http_client())
        parts = await collector.collect(request.image_urls, files)
        logger.debug("Calling %s with %d attachment(s)", self.get_webhook_url(), len(parts))
        return Payload(fields=fields, files=parts)

    def _session_id(self, request: ChatRequest, context: CallContext) -> str:
        base = self._session_strategies[self.valves.session_id_mode](request, context)
        # Background tasks (title, tags...) must not pollute the chat memory in n8n
        return f"{base}:{context.task}" if context.task else base

    def _chat_session(self, request: ChatRequest, context: CallContext) -> str:
        return context.chat_id or self._legacy_session(request, context)

    @staticmethod
    def _legacy_session(request: ChatRequest, context: CallContext) -> str:
        prefix = request.first_message_text[:LEGACY_SESSION_PREFIX_LENGTH]
        return f"{context.user_id} - {prefix}" if prefix else context.user_id

    def _get_http_client(self) -> httpx.AsyncClient:
        # Timeouts are passed per request so valve changes apply immediately (issue #7)
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient()
        return self._http_client

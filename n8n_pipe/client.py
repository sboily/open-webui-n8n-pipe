"""HTTP calls to the n8n webhook: retries, heartbeat, streaming and response parsing."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .attachments import FilePart
from .constants import (
    HTTP_ERROR_BODY_PREVIEW,
    JSON_CONTENT_TYPE,
    RETRY_BACKOFF_BASE_SECONDS,
    RETRY_BACKOFF_MAX_SECONDS,
    RETRYABLE_STATUS_CODES,
    StreamChunkType,
)
from .errors import N8nPipeError
from .messages import describe, t
from .status import StatusEmitter
from .valves import Valves

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.RemoteProtocolError)

Operation = Callable[[], Awaitable[httpx.Response]]


class _RetryableError(Exception):
    """Internal: the attempt failed in a way that allows another try."""


@dataclass(frozen=True)
class Payload:
    """What is sent to n8n: JSON when there is no attachment, multipart otherwise."""

    fields: dict[str, Any]
    files: list[FilePart] = field(default_factory=list)

    def request_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments for ``httpx`` (``json`` or ``data`` + ``files``)."""
        if not self.files:
            return {"json": self.fields}
        data = {
            key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            for key, value in self.fields.items()
        }
        return {"data": data, "files": self.files}


class N8nClient:
    """Send one payload to the n8n webhook and turn the answer into text."""

    def __init__(
        self, valves: Valves, http_client: httpx.AsyncClient, status: StatusEmitter, url: str
    ) -> None:
        self._valves = valves
        self._client = http_client
        self._status = status
        self._url = url
        self._chunk_handlers: dict[str, Callable[[dict[str, Any]], str | None]] = {
            StreamChunkType.ITEM.value: lambda chunk: str(chunk.get("content", "")),
            StreamChunkType.ERROR.value: self._raise_stream_error,
        }

    async def send(self, payload: Payload) -> str:
        """POST ``payload`` and return the text found in the configured response field."""
        response = await self._with_retry(lambda: self._await_with_heartbeat(self._post(payload)))
        return self._extract_output(self._parse_json(response))

    async def stream(self, payload: Payload) -> AsyncIterator[str]:
        """POST ``payload`` and yield the answer as it arrives.

        Handles both the n8n newline-delimited JSON streaming format and a
        classic JSON answer (workflow not configured for streaming).
        """
        response = await self._with_retry(lambda: self._open_stream(payload))
        try:
            if response.headers.get("content-type", "").startswith(JSON_CONTENT_TYPE):
                await response.aread()
                yield self._extract_output(self._parse_json(response))
                return
            await self._status.info("status.streaming")
            async for line in response.aiter_lines():
                text = self._parse_stream_line(line)
                if text:
                    yield text
        finally:
            await response.aclose()

    def _t(self, key: str, **params: object) -> str:
        return t(self._valves.language, key, **params)

    def _headers(self) -> dict[str, str]:
        token = self._valves.n8n_bearer_token
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def _post(self, payload: Payload) -> httpx.Response:
        return await self._client.post(
            self._url,
            headers=self._headers(),
            timeout=self._valves.http_timeout,
            **payload.request_kwargs(),
        )

    async def _open_stream(self, payload: Payload) -> httpx.Response:
        request = self._client.build_request(
            "POST",
            self._url,
            headers=self._headers(),
            timeout=self._valves.http_timeout,
            **payload.request_kwargs(),
        )
        return await self._client.send(request, stream=True)

    async def _with_retry(self, operation: Operation) -> httpx.Response:
        total = self._valves.max_retries + 1
        attempt = 0
        while True:
            attempt += 1
            await self._status.info("status.attempt", attempt=attempt, total=total)
            try:
                return await self._attempt(operation)
            except _RetryableError as failure:
                if attempt == total:
                    message = self._t("error.retries_exhausted", total=total, error=str(failure))
                    raise N8nPipeError(message) from failure
                await self._backoff(attempt, total)

    async def _attempt(self, operation: Operation) -> httpx.Response:
        try:
            response = await operation()
        except httpx.TimeoutException as error:
            raise N8nPipeError(self._t("error.timeout", seconds=self._valves.timeout)) from error
        except RETRYABLE_EXCEPTIONS as error:
            raise _RetryableError(describe(error)) from error
        except httpx.HTTPError as error:
            raise N8nPipeError(self._t("error.transport", error=describe(error))) from error
        if response.status_code in RETRYABLE_STATUS_CODES:
            await response.aclose()
            raise _RetryableError(f"HTTP {response.status_code}")
        if not response.is_success:
            await response.aread()
            body = response.text[:HTTP_ERROR_BODY_PREVIEW]
            raise N8nPipeError(self._t("error.http_status", status=response.status_code, body=body))
        return response

    async def _backoff(self, attempt: int, total: int) -> None:
        delay = min(RETRY_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), RETRY_BACKOFF_MAX_SECONDS)
        await self._status.warning("status.retry", delay=delay, attempt=attempt, total=total)
        await asyncio.sleep(delay)

    async def _await_with_heartbeat(self, coroutine: Awaitable[httpx.Response]) -> httpx.Response:
        """Await ``coroutine`` while emitting a status every ``emit_interval`` seconds."""
        task = asyncio.ensure_future(coroutine)
        started = time.monotonic()
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self._valves.emit_interval)
                if done:
                    return task.result()
                elapsed = int(time.monotonic() - started)
                await self._status.info("status.waiting", elapsed=elapsed)
        finally:
            if not task.done():
                task.cancel()

    def _parse_json(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise N8nPipeError(self._t("error.invalid_json")) from error

    def _extract_output(self, data: Any) -> str:
        # n8n "When Last Node Finishes / All Entries" wraps the answer in a list
        item = data[0] if isinstance(data, list) and data else data
        field_name = self._valves.response_field
        if not isinstance(item, dict) or field_name not in item:
            raise N8nPipeError(self._t("error.response_field", field=field_name))
        value = item[field_name]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def _parse_stream_line(self, line: str) -> str | None:
        if not line.strip():
            return None
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            return line  # plain-text stream
        if not isinstance(chunk, dict):
            return line
        handler = self._chunk_handlers.get(str(chunk.get("type")))
        return handler(chunk) if handler else None

    def _raise_stream_error(self, chunk: dict[str, Any]) -> str:
        raise N8nPipeError(self._t("error.stream", error=chunk.get("content", "")))

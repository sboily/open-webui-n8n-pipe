"""Attachments forwarded to n8n as multipart parts: message images and uploaded files."""

import asyncio
import base64
import binascii
import inspect
import ipaddress
import logging
import mimetypes
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from .constants import (
    ATTACHED_FILE_TYPE,
    BASE64_MARKER,
    BINARY_EXTENSION,
    BINARY_MIME,
    BYTES_PER_MB,
    DATA_URL_SCHEME,
    DEFAULT_IMAGE_MIME,
    DOWNLOAD_CHUNK_SIZE,
    FILE_PART_PREFIX,
    HTTP_SCHEMES,
    IMAGE_MIME_PREFIX,
    IMAGE_PART_PREFIX,
    TEXT_EXTENSION,
    TEXT_MIME,
)
from .errors import AttachmentError
from .messages import describe, t
from .valves import Valves

logger = logging.getLogger(__name__)

FilePart = tuple[str, tuple[str, bytes, str]]
"""httpx multipart entry: ``(field_name, (filename, content, mime_type))``."""

Loaded = tuple[bytes, str, str]
"""Content of an attachment: ``(content, mime_type, filename)``."""

FileLoader = Callable[[dict[str, Any]], Awaitable[Loaded | None]]


def extension_for(mime_type: str) -> str:
    """Return a safe file extension for ``mime_type`` (``.bin`` when unknown)."""
    return mimetypes.guess_extension(mime_type) or BINARY_EXTENSION


async def resolve_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve ``host`` to its IP addresses."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    return [ipaddress.ip_address(info[4][0]) for info in infos]


async def load_stored_file(record: dict[str, Any]) -> Loaded | None:
    """Read an uploaded file through Open-WebUI's storage layer.

    Returns ``None`` outside Open-WebUI (modules absent) or when the record has
    no storage path, so the caller can try another source.
    """
    file_id = record.get("id")
    if not file_id:
        return None
    try:
        from open_webui.models.files import Files
        from open_webui.storage.provider import Storage
    except ImportError:
        return None
    stored = Files.get_file_by_id(file_id)
    if inspect.isawaitable(stored):  # Open-WebUI made this call async in 2025
        stored = await stored
    path = getattr(stored, "path", None)
    if not path:
        return None
    local_path = await asyncio.to_thread(Storage.get_file, path)
    content = await asyncio.to_thread(Path(local_path).read_bytes)
    mime = (record.get("meta") or {}).get("content_type") or BINARY_MIME
    return content, mime, Path(record.get("filename") or file_id).name


async def load_extracted_text(record: dict[str, Any]) -> Loaded | None:
    """Use the text Open-WebUI extracted from the document when bytes are unavailable."""
    text = (record.get("data") or {}).get("content")
    if not text:
        return None
    stem = Path(record.get("filename") or record.get("id") or FILE_PART_PREFIX).stem
    return text.encode("utf-8"), TEXT_MIME, f"{stem}{TEXT_EXTENSION}"


FILE_LOADERS: tuple[FileLoader, ...] = (load_stored_file, load_extracted_text)


class AttachmentCollector:
    """Turn message images and ``__files__`` items into validated multipart parts."""

    def __init__(self, valves: Valves, http_client: httpx.AsyncClient) -> None:
        self._valves = valves
        self._client = http_client
        self._limit_bytes = valves.max_attachment_size_mb * BYTES_PER_MB
        self._image_loaders: dict[str, Callable[[str, str], Awaitable[tuple[bytes, str]]]] = {
            DATA_URL_SCHEME: self._load_data_url,
            **dict.fromkeys(HTTP_SCHEMES, self._load_remote_url),
        }

    async def collect(self, image_urls: list[str], files: list[dict[str, Any]]) -> list[FilePart]:
        """Return the multipart parts for ``image_urls`` and the uploaded ``files``."""
        parts = [await self._image_part(idx, url) for idx, url in enumerate(image_urls)]
        documents = [item for item in files if item.get("type") == ATTACHED_FILE_TYPE]
        parts.extend([await self._file_part(idx, item) for idx, item in enumerate(documents)])
        return parts

    def _t(self, key: str, **params: object) -> str:
        return t(self._valves.language, key, **params)

    def _check_size(self, size: int, name: str) -> None:
        if size > self._limit_bytes:
            limit = self._valves.max_attachment_size_mb
            raise AttachmentError(self._t("error.attachment_too_large", name=name, limit=limit))

    async def _image_part(self, idx: int, url: str) -> FilePart:
        name = f"{IMAGE_PART_PREFIX}{idx}"
        content, mime = await self._image_loader_for(url, name)(url, name)
        self._check_size(len(content), name)
        if not mime.startswith(IMAGE_MIME_PREFIX):
            raise AttachmentError(self._t("error.attachment_not_image", name=name, mime=mime))
        return name, (f"{name}{extension_for(mime)}", content, mime)

    def _image_loader_for(
        self, url: str, name: str
    ) -> Callable[[str, str], Awaitable[tuple[bytes, str]]]:
        for prefix, loader in self._image_loaders.items():
            if url.startswith(prefix):
                return loader
        raise AttachmentError(self._t("error.unsupported_url", name=name))

    async def _load_data_url(self, url: str, name: str) -> tuple[bytes, str]:
        header, marker, encoded = url.partition(BASE64_MARKER)
        if not marker:
            raise AttachmentError(self._t("error.invalid_data_url", name=name))
        mime = header[len(DATA_URL_SCHEME) :].split(";")[0] or DEFAULT_IMAGE_MIME
        # base64 inflates data by 4/3: refuse oversized payloads before decoding them
        self._check_size(len(encoded) * 3 // 4, name)
        try:
            return base64.b64decode(encoded, validate=True), mime
        except (binascii.Error, ValueError) as error:
            raise AttachmentError(self._t("error.invalid_data_url", name=name)) from error

    async def _load_remote_url(self, url: str, name: str) -> tuple[bytes, str]:
        if not self._valves.allow_remote_images:
            raise AttachmentError(self._t("error.remote_images_disabled"))
        await self._assert_public_host(url, name)
        try:
            async with self._client.stream(
                "GET", url, timeout=self._valves.http_timeout
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", DEFAULT_IMAGE_MIME)
                content = await self._read_bounded(response, name)
        except httpx.HTTPError as error:
            detail = describe(error)
            raise AttachmentError(
                self._t("error.remote_download", name=name, error=detail)
            ) from error
        return content, content_type.split(";")[0].strip()

    async def _assert_public_host(self, url: str, name: str) -> None:
        host = httpx.URL(url).host
        try:
            addresses = await resolve_addresses(host)
        except OSError as error:
            detail = describe(error)
            raise AttachmentError(
                self._t("error.remote_download", name=name, error=detail)
            ) from error
        if any(not address.is_global for address in addresses):
            raise AttachmentError(self._t("error.private_host", name=name))

    async def _read_bounded(self, response: httpx.Response, name: str) -> bytes:
        declared = response.headers.get("content-length", "")
        if declared.isdigit():
            self._check_size(int(declared), name)
        buffer = bytearray()
        async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_SIZE):
            buffer.extend(chunk)
            self._check_size(len(buffer), name)
        return bytes(buffer)

    async def _file_part(self, idx: int, item: dict[str, Any]) -> FilePart:
        record = item.get("file") or {}
        name = item.get("name") or record.get("filename") or f"{FILE_PART_PREFIX}{idx}"
        for loader in FILE_LOADERS:
            loaded = await loader(record)
            if loaded is None:
                continue
            content, mime, filename = loaded
            self._check_size(len(content), name)
            logger.debug("Attached %s via %s", filename, loader.__name__)
            return f"{FILE_PART_PREFIX}{idx}", (filename, content, mime)
        raise AttachmentError(self._t("error.file_unreadable", name=name))

"""Configuration of the pipe, edited by Open-WebUI administrators."""

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import (
    CONNECT_TIMEOUT_SECONDS,
    HTTP_SCHEMES,
    WEBHOOK_ID_PATTERN,
    Language,
    SessionIdMode,
)


class Valves(BaseModel):
    """Configuration parameters for the n8n pipe connector."""

    model_config = ConfigDict(validate_assignment=True)

    n8n_host: str = Field(
        default="http://localhost:5678",
        description="Base URL of the n8n server, e.g. https://n8n.example.com",
    )
    n8n_webhook_id: str = Field(
        default="your-webhook-id",
        description="Path of the n8n Webhook node (the part after /webhook/)",
    )
    n8n_test_mode: bool = Field(
        default=False,
        description="Call /webhook-test/ (n8n 'Test workflow' URL) instead of /webhook/",
    )
    n8n_bearer_token: str = Field(
        default="",
        description="Bearer token sent in the Authorization header (leave empty to send none)",
    )
    input_field: str = Field(
        default="chatInput",
        description="Name of the field carrying the user question in the payload sent to n8n",
    )
    response_field: str = Field(
        default="output",
        description="Name of the field carrying the answer in the JSON returned by n8n",
    )
    timeout: float = Field(
        default=120.0,
        gt=0,
        description="Seconds to wait for the n8n response (applied to every request)",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Retries on connection errors and HTTP 502/503/504 (never on timeouts)",
    )
    emit_interval: float = Field(
        default=2.0,
        gt=0,
        description="Seconds between 'waiting' status heartbeats while n8n is working",
    )
    enable_status_indicator: bool = Field(
        default=True,
        description="Emit status events (progress, retries, errors) to the chat UI",
    )
    session_id_mode: SessionIdMode = Field(
        default=SessionIdMode.CHAT_ID,
        description="sessionId strategy: 'chat_id' (stable per chat) or 'legacy' (user + message)",
    )
    stream_response: bool = Field(
        default=False,
        description="Read the n8n answer as a stream (Webhook node 'Response mode: Streaming')",
    )
    allow_remote_images: bool = Field(
        default=False,
        description="Allow downloading http(s) image URLs found in messages (SSRF risk)",
    )
    max_attachment_size_mb: int = Field(
        default=10,
        gt=0,
        description="Maximum size of a single attachment forwarded to n8n, in megabytes",
    )
    language: Language = Field(
        default=Language.EN,
        description="Language of status and error messages shown to users (en, fr)",
    )

    @field_validator("n8n_host")
    @classmethod
    def validate_host_url(cls, value: str) -> str:
        """Require an explicit http:// or https:// scheme."""
        if not value.startswith(HTTP_SCHEMES):
            raise ValueError("n8n_host must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("n8n_webhook_id")
    @classmethod
    def validate_webhook_id(cls, value: str) -> str:
        """Reject characters that could alter the webhook URL path."""
        if not WEBHOOK_ID_PATTERN.match(value):
            raise ValueError(
                "n8n_webhook_id must contain only alphanumeric characters, hyphens and underscores"
            )
        return value

    @property
    def http_timeout(self) -> httpx.Timeout:
        """Per-request timeout: short connect, ``timeout`` valve for read/write."""
        return httpx.Timeout(CONNECT_TIMEOUT_SECONDS, read=self.timeout, write=self.timeout)

"""Constants and enums shared by every module of the n8n pipe."""

import re
from enum import Enum


class StatusLevel(str, Enum):
    """Severity of a status event emitted to Open-WebUI."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Role(str, Enum):
    """Chat message roles used by Open-WebUI."""

    USER = "user"
    ASSISTANT = "assistant"


class ContentPartType(str, Enum):
    """Types of the parts found in a multi-modal message content list."""

    TEXT = "text"
    IMAGE_URL = "image_url"


class SessionIdMode(str, Enum):
    """Strategies used to derive the n8n ``sessionId``."""

    CHAT_ID = "chat_id"
    LEGACY = "legacy"


class Language(str, Enum):
    """Languages available for user-facing messages."""

    EN = "en"
    FR = "fr"


class StreamChunkType(str, Enum):
    """Chunk types of the n8n newline-delimited JSON streaming format."""

    BEGIN = "begin"
    ITEM = "item"
    END = "end"
    ERROR = "error"


# Open-WebUI event protocol
STATUS_EVENT_TYPE = "status"
ATTACHED_FILE_TYPE = "file"

# n8n webhook routing
WEBHOOK_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
WEBHOOK_PATH = "webhook"
WEBHOOK_TEST_PATH = "webhook-test"
HTTP_SCHEMES = ("http://", "https://")

# Payload sent to n8n
SESSION_FIELD = "sessionId"
METADATA_FIELD = "metadata"
IMAGE_PART_PREFIX = "image_"
FILE_PART_PREFIX = "file_"
ANONYMOUS_USER_ID = "anonymous"
LEGACY_SESSION_PREFIX_LENGTH = 100

# HTTP behaviour
CONNECT_TIMEOUT_SECONDS = 10.0
RETRY_BACKOFF_BASE_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 10.0
RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})
JSON_CONTENT_TYPE = "application/json"

# Attachments
DATA_URL_SCHEME = "data:"
BASE64_MARKER = ";base64,"
IMAGE_MIME_PREFIX = "image/"
DEFAULT_IMAGE_MIME = "image/png"
TEXT_MIME = "text/plain"
BINARY_EXTENSION = ".bin"
TEXT_EXTENSION = ".txt"
BYTES_PER_MB = 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 64 * 1024
BINARY_MIME = "application/octet-stream"
HTTP_ERROR_BODY_PREVIEW = 200

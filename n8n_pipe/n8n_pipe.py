"""N8N Pipe Function for OpenWebUI.

This module provides a connector between Open-WebUI and n8n workflows.

title: N8N Pipe Function
author: Sylvain BOILY (fork from https://openwebui.com/f/coleam/n8n_pipe)
author_url: https://github.com/sboily/open-webui-n8n-pipe
funding_url: https://github.com/sboily/open-webui-n8n-pipe
version: 0.2
"""

import asyncio
import base64
import io
import logging
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

# Add type stubs for missing libraries
try:
    import httpx
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    # For type checking only
    pass


# Configure logger
logger = logging.getLogger(__name__)

WEBHOOK_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class Pipe:
    """N8N Pipe connector for Open-WebUI.

    This class implements an asynchronous connector between Open-WebUI
    and n8n workflows for redirecting chat messages.
    """

    class Valves(BaseModel):
        """Configuration parameters for the N8N Pipe connector."""

        n8n_host: str = Field(
            default="http://localhost:5678",
            description="Base URL for n8n server (without trailing slash)",
        )
        n8n_webhook_id: str = Field(
            default="your-webhook-id",
            description="Webhook ID from n8n",
        )
        n8n_test_mode: bool = Field(
            default=False,
            description="Whether to use test mode URLs for n8n webhooks",
        )
        n8n_bearer_token: str = Field(
            default="your-token-here",
            description="Bearer token for n8n authentication",
        )
        input_field: str = Field(
            default="chatInput",
            description="Field name for the input in the JSON payload",
        )
        response_field: str = Field(
            default="output",
            description="Field name for the output in the response JSON",
        )
        emit_interval: float = Field(
            default=2.0,
            description="Interval in seconds between status emissions",
        )
        enable_status_indicator: bool = Field(
            default=True,
            description="Enable or disable status indicator emissions",
        )
        timeout: float = Field(
            default=30.0,
            description="Timeout for HTTP requests in seconds",
        )
        max_retries: int = Field(
            default=2,
            description="Maximum number of retries for failed requests",
        )
        history_limit: int = Field(
            default=10,
            description="Maximum number of messages to include in history",
        )

        @field_validator("n8n_host")
        @classmethod
        def validate_host_url(cls, v: str) -> str:
            """Validate that the n8n host URL includes a protocol.

            Args:
                v: The URL to validate

            Returns:
                The validated URL if it passes validation

            Raises:
                ValueError: If the URL does not start with http:// or https://
            """
            if not v.startswith(("http://", "https://")):
                raise ValueError("n8n_host must start with http:// or https://")
            return v

        @field_validator("n8n_webhook_id")
        @classmethod
        def validate_webhook_id(cls, v: str) -> str:
            """Validate that the webhook ID contains only safe characters.

            Args:
                v: The webhook ID to validate

            Returns:
                The validated webhook ID

            Raises:
                ValueError: If the webhook ID contains unsafe characters
            """
            if not WEBHOOK_ID_PATTERN.match(v):
                raise ValueError(
                    "n8n_webhook_id must contain only alphanumeric "
                    "characters, hyphens, and underscores"
                )
            return v

    def __init__(self) -> None:
        """Initialize the N8N Pipe connector."""
        self.type = "pipe"
        self.id = "n8n_pipe"
        self.name = "N8N Pipe"
        self.valves = self.Valves()
        self.last_emit_time: float = 0.0
        self._http_client: Optional[httpx.AsyncClient] = None
        logger.info("Initialized %s", self.name)

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with current timeout settings.

        Returns:
            An httpx.AsyncClient configured with the current timeout
        """
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.valves.timeout)
        return self._http_client

    async def __aenter__(self) -> "Pipe":
        """Async context manager entry method.

        Returns:
            Self, for use in async with statements
        """
        logger.debug("Entering async context")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit method, closes the HTTP client.

        Args:
            exc_type: Exception type, if an exception was raised
            exc_val: Exception value, if an exception was raised
            exc_tb: Exception traceback, if an exception was raised
        """
        logger.debug("Exiting async context")
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool,
    ) -> None:
        """Emit status update if conditions are met.

        Args:
            __event_emitter__: Callable to emit status events
            level: Status level (info, warning, error)
            message: Status description
            done: Whether this is a completion status
        """
        if not __event_emitter__:
            return

        current_time = time.time()
        if self.valves.enable_status_indicator and (
            current_time - self.last_emit_time >= self.valves.emit_interval or done
        ):
            logger.debug("Emitting status: %s - %s", level, message)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    def get_webhook_url(self) -> str:
        """Build the n8n webhook URL based on configuration settings.

        Returns:
            Complete webhook URL with appropriate path for test
            or production mode
        """
        base_url = self.valves.n8n_host.rstrip("/")
        webhook_id = self.valves.n8n_webhook_id
        path = (
            f"/webhook-test/{webhook_id}" if self.valves.n8n_test_mode else f"/webhook/{webhook_id}"
        )
        return urljoin(f"{base_url}/", path.lstrip("/"))

    def _extract_question(self, content: Union[str, list]) -> str:
        """Extract the actual question from the content.

        Args:
            content: The message content to process, either a string
                     or a list

        Returns:
            The cleaned question string
        """
        # Handle case when content is a list (e.g., when images are attached)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
            content = " ".join(text_parts)

        # Handle string content
        if isinstance(content, str):
            if "Prompt: " in content:
                return content.split("Prompt: ")[-1]
            return content

        return ""

    def _create_session_id(
        self,
        user: Optional[Dict[str, Any]],
        first_message: Optional[Union[str, list]],
    ) -> str:
        """Create a session identifier for the n8n workflow.

        Args:
            user: User information if available
            first_message: The first message in the conversation

        Returns:
            A session identifier string
        """
        user_id = user["id"] if user and "id" in user else "anonymous"
        message_prefix = ""

        if first_message:
            # Clean and truncate the first message
            clean_message = self._extract_question(first_message)[:100]
            message_prefix = f" - {clean_message}"

        return f"{user_id}{message_prefix}"

    async def _handle_http_error(
        self,
        error: Exception,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
    ) -> Dict[str, str]:
        """Handle HTTP and other errors uniformly.

        Args:
            error: The exception that was raised
            __event_emitter__: Callable to emit status events

        Returns:
            Error response dictionary
        """
        error_msg = str(error)
        logger.error("Error in n8n pipe: %s", error_msg)

        await self.emit_status(
            __event_emitter__,
            "error",
            f"Error during sequence execution: {error_msg}",
            True,
        )

        return {"error": error_msg}

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> Union[Dict[str, str], str, Any]:
        """Process the incoming request and return the response from N8N.

        Args:
            body: Request body containing messages
            __user__: User information (optional)
            __event_emitter__: Callable to emit status events
            __event_call__: Callable for event calls (reserved for
                Open-WebUI interface compatibility)

        Returns:
            N8N response or error dictionary
        """
        await self.emit_status(__event_emitter__, "info", "Calling N8N Workflow...", False)

        messages = body.get("messages", [])

        # Check if messages list is empty
        if not messages:
            await self.emit_status(
                __event_emitter__,
                "error",
                "No messages found in the request body",
                True,
            )

            # Add response message
            error_message = "No messages found in the request body"
            body["messages"] = body.get("messages", []) + [
                {"role": "assistant", "content": error_message}
            ]
            return {"error": error_message}

        # Extract the latest question
        last_message = messages[-1]
        if last_message.get("role") != "user":
            await self.emit_status(
                __event_emitter__,
                "error",
                "Last message is not from user",
                True,
            )
            error_message = "The last message must be from a user"
            body["messages"].append({"role": "assistant", "content": error_message})
            return {"error": error_message}

        # Extract content from the last message
        question_parts = last_message.get("content", "")
        question = self._extract_question(question_parts)

        # Extract image URLs if present
        image_urls = []
        if isinstance(question_parts, list):
            for item in question_parts:
                if (
                    item.get("type") == "image_url"
                    and "image_url" in item
                    and "url" in item["image_url"]
                ):
                    image_urls.append(item["image_url"]["url"])

        # If no question content found
        if not question.strip() and not image_urls:
            await self.emit_status(
                __event_emitter__,
                "warning",
                "Empty question received",
                True,
            )
            error_message = "Please provide a non-empty question"
            body["messages"].append({"role": "assistant", "content": error_message})
            return {"error": error_message}

        try:
            # Extract first message content for session ID
            first_message_content = ""
            if messages and messages[0].get("content"):
                first_message_content = messages[0]["content"]

            # Create session ID
            session_id = self._create_session_id(__user__, first_message_content)

            # Prepare the payload
            payload = {"sessionId": session_id}
            payload[self.valves.input_field] = question

            # Get headers
            headers = {
                "Authorization": (f"Bearer {self.valves.n8n_bearer_token}"),
                "Content-Type": "application/json",
            }

            # Get the appropriate webhook URL
            webhook_url = self.get_webhook_url()

            # Log message content type
            if image_urls:
                logger.info(
                    "Calling n8n webhook at: %s with text and %d image(s)",
                    webhook_url,
                    len(image_urls),
                )
            else:
                logger.info(
                    "Calling n8n webhook at: %s with text only",
                    webhook_url,
                )

            # Get HTTP client (lazy initialization)
            http_client = self._get_http_client()

            # Try request with retries
            n8n_response: Optional[str] = None
            retry_count = 0
            last_error: Optional[Exception] = None

            while retry_count <= self.valves.max_retries:
                try:
                    await self.emit_status(
                        __event_emitter__,
                        "info",
                        f"Attempt {retry_count + 1}/" f"{self.valves.max_retries + 1}",
                        False,
                    )

                    # Use httpx for async HTTP request
                    if image_urls:
                        response = await self._send_with_images(
                            http_client,
                            webhook_url,
                            session_id,
                            question,
                            image_urls,
                        )
                    else:
                        response = await http_client.post(
                            webhook_url,
                            json=payload,
                            headers=headers,
                        )

                    # Check status code
                    if response.status_code == 200:
                        response_json = response.json()
                        if self.valves.response_field in response_json:
                            n8n_response = response_json[self.valves.response_field]
                            break
                        else:
                            raise KeyError(
                                f"Response field "
                                f"'{self.valves.response_field}' "
                                "not found in N8N response"
                            )
                    else:
                        raise httpx.HTTPStatusError(
                            f"Error: {response.status_code}" f" - {response.text}",
                            request=response.request,
                            response=response,
                        )

                except (httpx.HTTPError, KeyError) as e:
                    last_error = e
                    retry_count += 1
                    if retry_count <= self.valves.max_retries:
                        await self.emit_status(
                            __event_emitter__,
                            "warning",
                            f"Retry {retry_count}/" f"{self.valves.max_retries}",
                            False,
                        )
                        await asyncio.sleep(1)

            # If we've exhausted retries
            if n8n_response is None and last_error is not None:
                return await self._handle_http_error(last_error, __event_emitter__)

            # Set assistant message with workflow reply
            body["messages"].append({"role": "assistant", "content": n8n_response})

            # Limit history if configured
            if 0 < self.valves.history_limit < len(body["messages"]):
                body["messages"] = body["messages"][-self.valves.history_limit :]

            await self.emit_status(__event_emitter__, "info", "Complete", True)
            return n8n_response

        except Exception as e:
            return await self._handle_http_error(e, __event_emitter__)

    async def _send_with_images(
        self,
        http_client: httpx.AsyncClient,
        webhook_url: str,
        session_id: str,
        question: str,
        image_urls: list,
    ) -> httpx.Response:
        """Send a multipart request with images to the webhook.

        Args:
            http_client: The HTTP client to use
            webhook_url: The webhook URL to send to
            session_id: The session identifier
            question: The text question
            image_urls: List of image URLs (data: or http(s)://)

        Returns:
            The HTTP response from n8n
        """
        form_data: Dict[str, str] = {}
        files: List[Tuple[str, Tuple[str, io.BytesIO, str]]] = []

        # Add text fields to form data
        form_data["sessionId"] = session_id
        form_data[self.valves.input_field] = question

        # Add image files from image_urls
        for idx, image_url in enumerate(image_urls):
            if image_url.startswith("data:"):
                self._process_data_url(image_url, idx, files)
            elif image_url.startswith(("http://", "https://")):
                await self._download_and_attach(http_client, image_url, idx, files)
            else:
                logger.warning(
                    "Unsupported image URL scheme for image_%d, " "skipping",
                    idx,
                )

        form_headers = {"Authorization": (f"Bearer {self.valves.n8n_bearer_token}")}

        return await http_client.post(
            webhook_url,
            data=form_data,
            files=files,
            headers=form_headers,
        )

    def _process_data_url(self, image_url: str, idx: int, files: list) -> None:
        """Process a data: URL and append to the files list.

        Args:
            image_url: The data: URL to process
            idx: The image index
            files: The files list to append to
        """
        mime_type = image_url.split(";")[0].split(":")[1]
        file_ext = ".jpg" if "jpeg" in mime_type else "." + mime_type.split("/")[-1]
        try:
            encoded_data = image_url.split(",")[1]
            image_data = base64.b64decode(encoded_data)
            files.append(
                (
                    f"image_{idx}",
                    (
                        f"image_{idx}{file_ext}",
                        io.BytesIO(image_data),
                        mime_type,
                    ),
                )
            )
            logger.debug("Added image_%d from data URL", idx)
        except Exception as e:
            logger.error("Failed to process image data URL: %s", str(e))

    async def _download_and_attach(
        self,
        http_client: httpx.AsyncClient,
        image_url: str,
        idx: int,
        files: list,
    ) -> None:
        """Download an image from a URL and attach it to the files list.

        Args:
            http_client: The HTTP client to use for downloading
            image_url: The HTTP(S) URL to download from
            idx: The image index
            files: The files list to append to
        """
        try:
            resp = await http_client.get(image_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/png")
            mime_type = content_type.split(";")[0].strip()
            file_ext = "." + mime_type.split("/")[-1]
            files.append(
                (
                    f"image_{idx}",
                    (
                        f"image_{idx}{file_ext}",
                        io.BytesIO(resp.content),
                        mime_type,
                    ),
                )
            )
            logger.debug("Added image_%d from remote URL", idx)
        except Exception as e:
            logger.error("Failed to download image from URL: %s", str(e))

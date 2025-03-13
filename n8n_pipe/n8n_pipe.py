"""N8N Pipe Function for OpenWebUI.

This module provides a connector between Open-WebUI and n8n workflows.

title: N8N Pipe Function
author: Sylvain BOILY (fork from https://openwebui.com/f/coleam/n8n_pipe)
author_url: https://github.com/sboily/open-webui-n8n-pipe
funding_url: https://github.com/sboily/open-webui-n8n-pipe
version: 0.2
"""

import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Union
from urllib.parse import urljoin

# Add type stubs for missing libraries
try:
    import httpx
    from pydantic import BaseModel, Field, validator
except ImportError:
    # For type checking only
    pass


# Configure logger
logger = logging.getLogger(__name__)


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
            default=False, description="Whether to use test mode URLs for n8n webhooks"  # NOQA
        )
        n8n_bearer_token: str = Field(
            default="your-token-here", description="Bearer token for n8n authentication"  # NOQA
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
            default=2.0, description="Interval in seconds between status emissions"  # NOQA
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"  # NOQA
        )
        timeout: float = Field(
            default=30.0, description="Timeout for HTTP requests in seconds"
        )  # NOQA
        max_retries: int = Field(
            default=2, description="Maximum number of retries for failed requests"  # NOQA
        )
        history_limit: int = Field(
            default=10, description="Maximum number of messages to include in history"  # NOQA
        )

        @validator("n8n_host")
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
                raise ValueError("n8n_host must start with http:// or https://")  # NOQA
            return v

    def __init__(self) -> None:
        """Initialize the N8N Pipe connector."""
        self.type = "pipe"
        self.id = "n8n_pipe"
        self.name = "N8N Pipe"
        self.valves = self.Valves()
        self.last_emit_time: float = 0.0
        self._http_client = httpx.AsyncClient(timeout=self.valves.timeout)
        logger.info(f"Initialized {self.name}")

    async def __aenter__(self) -> "Pipe":
        """Async context manager entry method.

        Returns:
            Self, for use in async with statements
        """
        logger.debug("Entering async context")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # NOQA
        """Async context manager exit method, closes the HTTP client.

        Args:
            exc_type: Exception type, if an exception was raised
            exc_val: Exception value, if an exception was raised
            exc_tb: Exception traceback, if an exception was raised
        """
        logger.debug("Exiting async context")
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
            current_time - self.last_emit_time >= self.valves.emit_interval or done  # NOQA
        ):
            logger.debug(f"Emitting status: {level} - {message}")
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
            f"/webhook-test/{webhook_id}"
            if self.valves.n8n_test_mode
            else f"/webhook/{webhook_id}"  # NOQA
        )
        return urljoin(f"{base_url}/", path.lstrip("/"))

    def _extract_question(self, content: str) -> str:
        """Extract the actual question from the content string.

        Args:
            content: The message content to process

        Returns:
            The cleaned question string
        """
        return content.split("Prompt: ")[-1] if "Prompt: " in content else content  # NOQA

    def _create_session_id(
        self, user: Optional[Dict[str, Any]], first_message: Optional[str]
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
        logger.error(f"Error in n8n pipe: {error_msg}")

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
            __event_call__: Callable for event calls

        Returns:
            N8N response or error dictionary
        """
        await self.emit_status(__event_emitter__, "info", "Calling N8N Workflow...", False)  # NOQA

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
            body["messages"].append({"role": "assistant", "content": error_message})  # NOQA
            return {"error": error_message}

        question = last_message.get("content", "")
        question = self._extract_question(question)

        # If no question content found
        if not question.strip():
            await self.emit_status(
                __event_emitter__,
                "warning",
                "Empty question received",
                True,
            )
            error_message = "Please provide a non-empty question"
            body["messages"].append({"role": "assistant", "content": error_message})  # NOQA
            return {"error": error_message}

        try:
            # Extract first message content for session ID
            first_message_content = ""
            if messages and messages[0].get("content"):
                first_message_content = messages[0]["content"]

            # Create session ID
            session_id = self._create_session_id(__user__, first_message_content)  # NOQA

            # Prepare the payload
            payload = {"sessionId": session_id}
            payload[self.valves.input_field] = question

            # Get headers
            headers = {
                "Authorization": f"Bearer {self.valves.n8n_bearer_token}",
                "Content-Type": "application/json",
            }

            # Get the appropriate webhook URL
            webhook_url = self.get_webhook_url()
            logger.info(f"Calling n8n webhook at: {webhook_url}")

            # Try request with retries
            n8n_response: Optional[str] = None
            retry_count = 0
            last_error: Optional[Exception] = None

            while retry_count <= self.valves.max_retries:
                try:
                    await self.emit_status(
                        __event_emitter__,
                        "info",
                        f"Attempt {retry_count + 1}/{self.valves.max_retries + 1}",  # NOQA
                        False,
                    )

                    # Use httpx for async HTTP request
                    response = await self._http_client.post(
                        webhook_url, json=payload, headers=headers
                    )

                    # Check status code
                    if response.status_code == 200:
                        response_json = response.json()
                        if self.valves.response_field in response_json:
                            n8n_response = response_json[self.valves.response_field]  # NOQA
                            break
                        else:
                            raise KeyError(
                                f"Response field '{self.valves.response_field}'"  # NOQA
                                " not found in N8N response"
                            )
                    else:
                        raise httpx.HTTPStatusError(
                            f"Error: {response.status_code} - {response.text}",
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
                            f"Retry {retry_count}/{self.valves.max_retries}",
                            False,
                        )
                        time.sleep(1)  # Simple backoff

            # If we've exhausted retries
            if n8n_response is None and last_error is not None:
                return await self._handle_http_error(last_error, __event_emitter__)  # NOQA

            # Set assistant message with workflow reply
            body["messages"].append({"role": "assistant", "content": n8n_response})  # NOQA

            # Limit history if configured
            if (
                self.valves.history_limit > 0
                and len(body["messages"]) > self.valves.history_limit  # NOQA
            ):  # NOQA
                body["messages"] = body["messages"][-self.valves.history_limit :]  # NOQA

            await self.emit_status(__event_emitter__, "info", "Complete", True)
            return n8n_response

        except Exception as e:
            return await self._handle_http_error(e, __event_emitter__)

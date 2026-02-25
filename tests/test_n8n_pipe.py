import base64
import io
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from n8n_pipe.n8n_pipe import Pipe


@pytest.fixture
def pipe():
    pipe = Pipe()
    # Configure test properties
    pipe.valves.n8n_host = "http://test-n8n.local:5678"
    pipe.valves.n8n_webhook_id = "test-webhook-id"
    pipe.valves.n8n_test_mode = False
    pipe.valves.max_retries = 1
    return pipe


@pytest.fixture
def event_emitter():
    return AsyncMock()


@pytest.fixture
def user():
    return {"id": "test_user_123"}


@pytest.fixture
def request_body():
    return {
        "messages": [
            {"role": "user", "content": "Prompt: Initial message"},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "Test question"},
        ]
    }


@pytest.mark.asyncio
async def test_emit_status(pipe, event_emitter):
    """Test the emit_status method with various conditions."""
    # Test normal emission
    pipe.last_emit_time = time.time() - pipe.valves.emit_interval - 1
    await pipe.emit_status(event_emitter, "info", "Test message", False)

    event_emitter.assert_called_once_with(
        {
            "type": "status",
            "data": {
                "status": "in_progress",
                "level": "info",
                "description": "Test message",
                "done": False,
            },
        }
    )

    # Reset mock and test emission before interval
    event_emitter.reset_mock()
    pipe.last_emit_time = time.time()
    await pipe.emit_status(event_emitter, "info", "Test message 2", False)

    # Should not emit because interval has not passed
    event_emitter.assert_not_called()

    # Test forced emission with done=True
    event_emitter.reset_mock()
    await pipe.emit_status(event_emitter, "info", "Test message 3", True)

    # Should emit regardless of interval because done=True
    event_emitter.assert_called_once()

    # Test disabled status indicator
    event_emitter.reset_mock()
    pipe.valves.enable_status_indicator = False
    await pipe.emit_status(event_emitter, "info", "Test message 4", True)

    # Should not emit because status indicator is disabled
    event_emitter.assert_not_called()

    # Test with None event_emitter
    event_emitter.reset_mock()
    pipe.valves.enable_status_indicator = True
    await pipe.emit_status(None, "info", "Test message 5", True)

    # Should handle None event_emitter gracefully
    event_emitter.assert_not_called()


@pytest.mark.asyncio
async def test_get_webhook_url(pipe):
    """Test the webhook URL generation for both production and test modes."""
    # Test production URL (default)
    pipe.valves.n8n_host = "http://n8n.example.com:5678"
    pipe.valves.n8n_webhook_id = "abc123"
    pipe.valves.n8n_test_mode = False

    assert pipe.get_webhook_url() == "http://n8n.example.com:5678/webhook/abc123"

    # Test test mode URL
    pipe.valves.n8n_test_mode = True
    assert pipe.get_webhook_url() == "http://n8n.example.com:5678/webhook-test/abc123"

    # Test with trailing slash in host URL
    pipe.valves.n8n_host = "http://n8n.example.com:5678/"
    assert pipe.get_webhook_url() == "http://n8n.example.com:5678/webhook-test/abc123"


@pytest.mark.asyncio
async def test_extract_question():
    """Test the _extract_question method."""
    pipe = Pipe()

    # Test with no prompt prefix
    assert pipe._extract_question("Simple question") == "Simple question"

    # Test with prompt prefix
    assert pipe._extract_question("Prompt: Actual question") == "Actual question"

    # Test with multiple prompt prefixes (should extract after the last one)
    assert pipe._extract_question("Prompt: First Prompt: Actual question") == "Actual question"


@pytest.mark.asyncio
async def test_create_session_id():
    """Test the _create_session_id method."""
    pipe = Pipe()

    # Test with user and message
    user = {"id": "user123"}
    message = "Test message"
    session_id = pipe._create_session_id(user, message)
    assert session_id == "user123 - Test message"

    # Test with user and prompt message
    user = {"id": "user123"}
    message = "Prompt: Test message"
    session_id = pipe._create_session_id(user, message)
    assert session_id == "user123 - Test message"

    # Test with long message (should truncate)
    user = {"id": "user123"}
    message = "A" * 200
    session_id = pipe._create_session_id(user, message)
    assert len(session_id) < 150  # Should truncate

    # Test without user
    user = None
    message = "Test message"
    session_id = pipe._create_session_id(user, message)
    assert session_id == "anonymous - Test message"

    # Test without message
    user = {"id": "user123"}
    message = None
    session_id = pipe._create_session_id(user, message)
    assert session_id == "user123"


@pytest.mark.asyncio
async def test_pipe_success(pipe, request_body, user, event_emitter):
    """Test successful execution of the pipe method."""
    expected_url = "http://test-n8n.local:5678/webhook/test-webhook-id"

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={pipe.valves.response_field: "Test N8N response"})

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    mock_client.post.assert_called_once()

    args, kwargs = mock_client.post.call_args
    assert args[0] == expected_url
    assert result == "Test N8N response"

    assert request_body["messages"][-1]["role"] == "assistant"
    assert request_body["messages"][-1]["content"] == "Test N8N response"


@pytest.mark.asyncio
async def test_pipe_http_error(pipe, request_body, user, event_emitter):
    """Test HTTP error handling in the pipe method."""
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_response.request = MagicMock()

    async def mock_post_async(*args, **kwargs):
        error = httpx.HTTPStatusError(
            f"Error: {mock_response.status_code} - {mock_response.text}",
            request=mock_response.request,
            response=mock_response,
        )
        raise error

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    assert isinstance(result, dict)
    assert "error" in result
    assert "404" in result["error"]

    error_found = False
    for call_args in event_emitter.call_args_list:
        args = call_args[0]
        if len(args) > 0:
            data = args[0]
            if (
                isinstance(data, dict)
                and data.get("type") == "status"
                and data.get("data", {}).get("level") == "error"
                and "404" in data.get("data", {}).get("description", "")
            ):
                error_found = True
                break

    assert error_found, "No error message about 404 error was emitted"


@pytest.mark.asyncio
async def test_pipe_response_field_missing(pipe, request_body, user, event_emitter):
    """Test handling of missing response field in the n8n response."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"wrong_field": "Test response"})

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    assert isinstance(result, dict)
    assert "error" in result
    assert "Response field" in result["error"]

    error_found = False
    for call_args in event_emitter.call_args_list:
        args = call_args[0]
        if len(args) > 0:
            data = args[0]
            if (
                isinstance(data, dict)
                and data.get("type") == "status"
                and data.get("data", {}).get("level") == "error"
                and "Response field" in data.get("data", {}).get("description", "")
            ):
                error_found = True
                break

    assert error_found, "No error message about the response field was emitted"


@pytest.mark.asyncio
async def test_pipe_exception(pipe, request_body, user, event_emitter):
    """Test general exception handling in the pipe method."""

    async def mock_error(*args, **kwargs):
        raise Exception("Test error")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_error)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    assert isinstance(result, dict)
    assert result["error"] == "Test error"

    error_found = False
    for call_args in event_emitter.call_args_list:
        args = call_args[0]
        if len(args) > 0:
            data = args[0]
            if (
                isinstance(data, dict)
                and data.get("type") == "status"
                and data.get("data", {}).get("level") == "error"
                and "Test error" in data.get("data", {}).get("description", "")
            ):
                error_found = True
                break

    assert error_found, "No error message about the exception was emitted"


@pytest.mark.asyncio
async def test_pipe_no_messages(pipe, event_emitter):
    """Test handling of empty messages array."""
    empty_body = {"messages": []}

    result = await pipe.pipe(empty_body, {"id": "test_user"}, event_emitter)

    assert empty_body["messages"][-1]["role"] == "assistant"
    assert "No messages found" in empty_body["messages"][-1]["content"]

    error_found = False
    for call_args in event_emitter.call_args_list:
        args = call_args[0]
        if len(args) > 0:
            data = args[0]
            if (
                isinstance(data, dict)
                and data.get("type") == "status"
                and data.get("data", {}).get("level") == "error"
                and "No messages found" in data.get("data", {}).get("description", "")
            ):
                error_found = True
                break

    assert error_found, "No error message about empty messages was emitted"


@pytest.mark.asyncio
async def test_pipe_empty_message(pipe, event_emitter):
    """Test handling of empty message content."""
    body = {"messages": [{"role": "user", "content": ""}]}

    result = await pipe.pipe(body, {"id": "test_user"}, event_emitter)

    assert body["messages"][-1]["role"] == "assistant"
    assert "non-empty question" in body["messages"][-1]["content"]

    warning_found = False
    for call_args in event_emitter.call_args_list:
        args = call_args[0]
        if len(args) > 0:
            data = args[0]
            if (
                isinstance(data, dict)
                and data.get("type") == "status"
                and data.get("data", {}).get("level") == "warning"
                and "Empty question" in data.get("data", {}).get("description", "")
            ):
                warning_found = True
                break

    assert warning_found, "No warning message about empty question was emitted"


@pytest.mark.asyncio
async def test_pipe_last_message_not_from_user(pipe, event_emitter):
    """Test handling when last message is not from user."""
    body = {
        "messages": [
            {"role": "user", "content": "Initial message"},
            {"role": "assistant", "content": "How can I help?"},
        ]
    }

    result = await pipe.pipe(body, {"id": "test_user"}, event_emitter)

    assert body["messages"][-1]["role"] == "assistant"
    assert "from a user" in body["messages"][-1]["content"]

    error_found = False
    for call_args in event_emitter.call_args_list:
        args = call_args[0]
        if len(args) > 0:
            data = args[0]
            if (
                isinstance(data, dict)
                and data.get("type") == "status"
                and data.get("data", {}).get("level") == "error"
                and "Last message is not from user" in data.get("data", {}).get("description", "")
            ):
                error_found = True
                break

    assert error_found, "No error message about last message not from user was emitted"


@pytest.mark.asyncio
async def test_pipe_test_mode(pipe, request_body, event_emitter):
    """Test the pipe with test mode enabled."""
    pipe.valves.n8n_test_mode = True

    expected_url = "http://test-n8n.local:5678/webhook-test/test-webhook-id"

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={pipe.valves.response_field: "Test mode response"})

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, None, event_emitter)

    args, kwargs = mock_client.post.call_args
    assert args[0] == expected_url
    assert "anonymous" in kwargs["json"]["sessionId"]
    assert result == "Test mode response"


@pytest.mark.asyncio
async def test_pipe_prompt_extraction(pipe, event_emitter):
    """Test extraction of content after 'Prompt: ' prefix."""
    body = {
        "messages": [
            {"role": "user", "content": "Prompt: Initial setup"},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "Prompt: Actual question about n8n"},
        ]
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={pipe.valves.response_field: "Answer about n8n"})

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(body, {"id": "test_user"}, event_emitter)

    args, kwargs = mock_client.post.call_args
    assert kwargs["json"][pipe.valves.input_field] == "Actual question about n8n"
    assert kwargs["json"]["sessionId"] == "test_user - Initial setup"


@pytest.mark.asyncio
async def test_retry_mechanism(pipe, request_body, user, event_emitter):
    """Test the retry mechanism for failed requests."""
    failure_response = AsyncMock()
    failure_response.status_code = 503
    failure_response.text = "Service Unavailable"
    failure_response.request = MagicMock()

    success_response = AsyncMock()
    success_response.status_code = 200
    success_response.json = MagicMock(
        return_value={pipe.valves.response_field: "Retry success response"}
    )

    side_effects = [
        httpx.HTTPStatusError(
            "Error: 503 - Service Unavailable",
            request=failure_response.request,
            response=failure_response,
        ),
        success_response,
    ]

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=side_effects)
    pipe._http_client = mock_client
    pipe.valves.max_retries = 1

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await pipe.pipe(request_body, user, event_emitter)

        assert mock_client.post.call_count == 2
        mock_sleep.assert_called_once_with(1)
        assert result == "Retry success response"


@pytest.mark.asyncio
async def test_history_limit(pipe, request_body, user, event_emitter):
    """Test the history limit functionality."""
    pipe.valves.history_limit = 3

    request_body["messages"] = [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user", "content": "Final question"},
    ]

    original_length = len(request_body["messages"])

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={pipe.valves.response_field: "Final answer"})

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    assert (
        len(request_body["messages"]) == pipe.valves.history_limit
    ), f"Expected {pipe.valves.history_limit} messages, got {len(request_body['messages'])}"
    assert len(request_body["messages"]) < original_length, "Message history was not trimmed"
    assert (
        request_body["messages"][-1]["content"] == "Final answer"
    ), "Final answer should be the last message"
    assert "Final question" in [
        msg["content"] for msg in request_body["messages"]
    ], "Last user question should be retained"


@pytest.mark.asyncio
async def test_async_context_manager():
    """Test the async context manager functionality."""

    async def mock_aclose():
        pass

    with patch("httpx.AsyncClient.aclose", new=AsyncMock(side_effect=mock_aclose)) as mock_close:
        async with Pipe() as pipe:
            assert isinstance(pipe, Pipe)
            # Force client creation so __aexit__ has something to close
            pipe._http_client = httpx.AsyncClient()

        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_validator_host_url():
    """Test the validator for n8n_host URL."""
    # Test valid URL at construction time
    from n8n_pipe.n8n_pipe import Pipe as PipeCls

    valves = PipeCls.Valves(n8n_host="https://example.com")
    assert valves.n8n_host == "https://example.com"

    # Test invalid URL at construction time raises ValueError
    with pytest.raises(Exception):
        PipeCls.Valves(n8n_host="example.com")


@pytest.mark.asyncio
async def test_validator_webhook_id():
    """Test the validator for n8n_webhook_id."""
    from n8n_pipe.n8n_pipe import Pipe as PipeCls

    # Valid webhook IDs
    valves = PipeCls.Valves(n8n_webhook_id="abc-123_test")
    assert valves.n8n_webhook_id == "abc-123_test"

    # Invalid webhook ID with path traversal
    with pytest.raises(Exception):
        PipeCls.Valves(n8n_webhook_id="../../admin/api")

    # Invalid webhook ID with special characters
    with pytest.raises(Exception):
        PipeCls.Valves(n8n_webhook_id="webhook id with spaces")


# Attachment tests


@pytest.mark.asyncio
async def test_extract_question_with_list_content():
    """Test _extract_question method with list content (images + text)."""
    pipe = Pipe()

    # Test with mixed content (text + image)
    list_content = [
        {"type": "text", "text": "What do you see in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAY"
                "AAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            },
        },
    ]

    result = pipe._extract_question(list_content)
    assert result == "What do you see in this image?"

    # Test with multiple text parts
    list_content = [
        {"type": "text", "text": "First part"},
        {"type": "text", "text": "Second part"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
    ]

    result = pipe._extract_question(list_content)
    assert result == "First part Second part"

    # Test with empty list
    result = pipe._extract_question([])
    assert result == ""

    # Test with only images (no text)
    list_content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}]

    result = pipe._extract_question(list_content)
    assert result == ""

    # Test with prompt prefix in list content
    list_content = [
        {"type": "text", "text": "Prompt: What do you see?"},
    ]

    result = pipe._extract_question(list_content)
    assert result == "What do you see?"


@pytest.mark.asyncio
async def test_create_session_id_with_list_content():
    """Test _create_session_id method with list content."""
    pipe = Pipe()

    user = {"id": "user123"}

    message_list = [
        {"type": "text", "text": "Initial message"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
    ]

    session_id = pipe._create_session_id(user, message_list)
    assert session_id.startswith("user123")


@pytest.mark.asyncio
async def test_pipe_with_image_attachments(pipe, user, event_emitter):
    """Test pipe method with image attachments."""
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA"
                            "EAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7w"
                            "AAAABJ RU5ErkJggg=="
                        },
                    },
                ],
            }
        ]
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={pipe.valves.response_field: "I see a small test image"}
    )

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    mock_client.post.assert_called_once()

    args, kwargs = mock_client.post.call_args
    assert "data" in kwargs
    assert "files" in kwargs
    assert "json" not in kwargs

    assert kwargs["data"]["sessionId"] is not None
    assert kwargs["data"][pipe.valves.input_field] == "What do you see in this image?"

    assert result == "I see a small test image"


@pytest.mark.asyncio
async def test_pipe_with_multiple_image_attachments(pipe, user, event_emitter):
    """Test pipe method with multiple image attachments."""
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA"
                            "EAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7w"
                            "AAAABJ RU5ErkJggg=="
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABg"
                            "AAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
                        },
                    },
                ],
            }
        ]
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={pipe.valves.response_field: "I see two test images"}
    )

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    args, kwargs = mock_client.post.call_args
    assert "files" in kwargs

    assert kwargs["data"][pipe.valves.input_field] == "Compare these images"
    assert result == "I see two test images"


@pytest.mark.asyncio
async def test_pipe_with_image_only_content(pipe, user, event_emitter):
    """Test pipe method with image-only content (no text)."""
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA"
                            "EAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7w"
                            "AAAABJ RU5ErkJggg=="
                        },
                    }
                ],
            }
        ]
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={pipe.valves.response_field: "Image analysis complete"}
    )

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    args, kwargs = mock_client.post.call_args
    assert "files" in kwargs
    assert kwargs["data"][pipe.valves.input_field] == ""
    assert result == "Image analysis complete"


@pytest.mark.asyncio
async def test_pipe_empty_content_no_images(pipe, event_emitter):
    """Test pipe method with empty content and no images (should fail)."""
    request_body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "   "}]}]}

    result = await pipe.pipe(request_body, None, event_emitter)

    assert isinstance(result, dict)
    assert "error" in result
    assert "Please provide a non-empty question" in result["error"]


@pytest.mark.asyncio
async def test_pipe_invalid_image_data(pipe, user, event_emitter):
    """Test pipe method with invalid base64 image data."""
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,invalid_base64_data!!!"},
                    },
                ],
            }
        ]
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={pipe.valves.response_field: "Text processed successfully"}
    )

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    with patch("n8n_pipe.n8n_pipe.logger") as mock_logger:
        result = await pipe.pipe(request_body, user, event_emitter)

        mock_logger.error.assert_called()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "Failed to process image data URL" in error_call_args

        assert result == "Text processed successfully"


@pytest.mark.asyncio
async def test_pipe_mixed_content_types(pipe, user, event_emitter):
    """Test pipe method with mixed content types in correct order."""
    request_body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA"
                            "EAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7w"
                            "AAAABJ RU5ErkJggg=="
                        },
                    },
                    {"type": "text", "text": "and tell me what you see"},
                ],
            },
        ]
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(
        return_value={pipe.valves.response_field: "I see a test pattern"}
    )

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    result = await pipe.pipe(request_body, user, event_emitter)

    args, kwargs = mock_client.post.call_args
    assert "files" in kwargs

    expected_text = "Analyze this image and tell me what you see"
    assert kwargs["data"][pipe.valves.input_field] == expected_text
    assert result == "I see a test pattern"


@pytest.mark.asyncio
async def test_attachment_logging(pipe, user, event_emitter):
    """Test that attachment presence is properly logged."""
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Process this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA"
                            "EAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7w"
                            "AAAABJ RU5ErkJggg=="
                        },
                    },
                ],
            }
        ]
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={pipe.valves.response_field: "Processed"})

    async def mock_post_async(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=mock_post_async)
    pipe._http_client = mock_client

    with patch("n8n_pipe.n8n_pipe.logger") as mock_logger:
        result = await pipe.pipe(request_body, user, event_emitter)

        mock_logger.info.assert_called()
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        image_log_found = any("with text and %d image(s)" in c for c in info_calls)
        assert image_log_found, f"Expected image logging not found in: {info_calls}"


@pytest.mark.asyncio
async def test_get_http_client_creates_new_when_none():
    """Test that _get_http_client creates a client when none exists."""
    pipe = Pipe()
    pipe._http_client = None
    client = pipe._get_http_client()
    assert client is not None
    assert not client.is_closed
    await client.aclose()


@pytest.mark.asyncio
async def test_get_http_client_creates_new_when_closed():
    """Test that _get_http_client creates a new client when existing is closed."""
    pipe = Pipe()
    client = pipe._get_http_client()
    await client.aclose()
    new_client = pipe._get_http_client()
    assert new_client is not None
    assert not new_client.is_closed
    await new_client.aclose()

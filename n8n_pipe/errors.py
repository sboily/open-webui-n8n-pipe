"""Exceptions raised by the n8n pipe.

Their message is already user-facing (translated), so Open-WebUI can display
it as-is when the exception propagates out of ``Pipe.pipe``.
"""


class N8nPipeError(Exception):
    """A failure that must be reported to the chat user."""


class AttachmentError(N8nPipeError):
    """An attachment could not be validated or loaded."""

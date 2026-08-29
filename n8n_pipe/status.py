"""Status events sent to the Open-WebUI chat interface."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .constants import STATUS_EVENT_TYPE, StatusLevel
from .messages import t
from .valves import Valves

logger = logging.getLogger(__name__)

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]


class StatusEmitter:
    """Translate and emit status events for one ``pipe`` call."""

    def __init__(self, emitter: EventEmitter | None, valves: Valves) -> None:
        """Wrap the Open-WebUI ``__event_emitter__`` (``None`` when absent)."""
        self._emitter = emitter
        self._valves = valves

    async def info(self, key: str, **params: object) -> None:
        """Emit an in-progress informational status."""
        await self._emit(StatusLevel.INFO, t(self._valves.language, key, **params), done=False)

    async def warning(self, key: str, **params: object) -> None:
        """Emit an in-progress warning status."""
        await self._emit(StatusLevel.WARNING, t(self._valves.language, key, **params), done=False)

    async def done(self, key: str, **params: object) -> None:
        """Emit the final successful status."""
        await self._emit(StatusLevel.INFO, t(self._valves.language, key, **params), done=True)

    async def error(self, message: str) -> None:
        """Emit the final error status with an already translated ``message``."""
        await self._emit(StatusLevel.ERROR, message, done=True)

    async def _emit(self, level: StatusLevel, description: str, done: bool) -> None:
        if self._emitter is None or not self._valves.enable_status_indicator:
            return
        logger.debug("Status %s: %s", level.value, description)
        await self._emitter(
            {
                "type": STATUS_EVENT_TYPE,
                "data": {"level": level.value, "description": description, "done": done},
            }
        )

from typing import Callable, Dict, List, Any
import asyncio
import logging

logger = logging.getLogger(__name__)

class CallbackManager:
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {}

    def register(self, event_name: str, callback: Callable):
        """Register a callback for a specific event."""
        if event_name not in self._callbacks:
            self._callbacks[event_name] = []
        self._callbacks[event_name].append(callback)
        logger.info(f"Registered callback for event: {event_name}")

    async def trigger(self, event_name: str, *args, **kwargs):
        """Trigger all callbacks registered for a specific event."""
        if event_name in self._callbacks:
            logger.debug(f"Triggering event: {event_name}")
            # Create a list of coroutines to run
            tasks = [callback(*args, **kwargs) for callback in self._callbacks[event_name]]
            # Run them concurrently
            await asyncio.gather(*tasks)

# Create a singleton instance
callback_manager = CallbackManager() 
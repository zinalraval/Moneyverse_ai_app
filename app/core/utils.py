import asyncio
import logging

logger = logging.getLogger(__name__)

def safe_create_task(coro):
    """
    Safely schedule an async coroutine as a background task.
    If called from a running event loop, uses create_task.
    If called from sync code, uses asyncio.run.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        # No running event loop, run synchronously (for scripts/tests)
        logger.warning("No running event loop, running coroutine synchronously. This is not recommended for production.")
        return asyncio.run(coro) 
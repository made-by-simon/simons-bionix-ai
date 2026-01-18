"""Keep-alive module using an async background task.

This approach is more reliable than relying solely on external pings:
1. Uses Discord's built-in event loop for the ping task
2. Pings the local web server to keep it responsive
3. Logs activity to show the bot is alive
"""

import asyncio
import aiohttp
from datetime import datetime

# Configuration.
PING_INTERVAL = 270  # 4.5 minutes (Replit sleeps after ~5 min of inactivity).
HEALTH_ENDPOINT = "http://localhost:8080/health"

_keep_alive_task = None


async def _ping_loop():
    """Background task that pings the health endpoint periodically."""
    await asyncio.sleep(30)  # Initial delay to let server start.

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(HEALTH_ENDPOINT, timeout=10) as response:
                    if response.status == 200:
                        print(f"[{datetime.now()}] Keep-alive ping successful")
                    else:
                        print(f"[{datetime.now()}] Keep-alive ping returned status {response.status}")
            except asyncio.TimeoutError:
                print(f"[{datetime.now()}] Keep-alive ping timed out")
            except aiohttp.ClientError as e:
                print(f"[{datetime.now()}] Keep-alive ping failed: {e}")
            except Exception as e:
                print(f"[{datetime.now()}] Keep-alive unexpected error: {e}")

            await asyncio.sleep(PING_INTERVAL)


def start(loop):
    """Start the keep-alive task on the given event loop.

    Args:
        loop: The asyncio event loop to schedule the task on.

    Returns:
        The created asyncio task.
    """
    global _keep_alive_task
    _keep_alive_task = loop.create_task(_ping_loop())
    print(f"[{datetime.now()}] Keep-alive task started (interval: {PING_INTERVAL}s)")
    return _keep_alive_task


def stop():
    """Stop the keep-alive task if running."""
    global _keep_alive_task
    if _keep_alive_task and not _keep_alive_task.done():
        _keep_alive_task.cancel()
        print(f"[{datetime.now()}] Keep-alive task stopped")
        _keep_alive_task = None

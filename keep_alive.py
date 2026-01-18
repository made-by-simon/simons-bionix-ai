"""Keep-alive module using an async background task."""

import asyncio
from datetime import datetime

import aiohttp

PING_INTERVAL = 270  # 4.5 minutes (Replit sleeps after ~5 min of inactivity).
HEALTH_ENDPOINT = "http://localhost:8080/health"

_task = None


async def _ping_loop():
    """Background task that pings the health endpoint periodically."""
    await asyncio.sleep(30)
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(HEALTH_ENDPOINT, timeout=10) as resp:
                    status = "successful" if resp.status == 200 else f"returned status {resp.status}"
                    print(f"[{datetime.now()}] Keep-alive ping {status}")
            except asyncio.TimeoutError:
                print(f"[{datetime.now()}] Keep-alive ping timed out")
            except aiohttp.ClientError as e:
                print(f"[{datetime.now()}] Keep-alive ping failed: {e}")
            except Exception as e:
                print(f"[{datetime.now()}] Keep-alive unexpected error: {e}")
            await asyncio.sleep(PING_INTERVAL)


def start(loop):
    """Start the keep-alive task on the given event loop."""
    global _task
    _task = loop.create_task(_ping_loop())
    print(f"[{datetime.now()}] Keep-alive task started (interval: {PING_INTERVAL}s)")
    return _task


def stop():
    """Stop the keep-alive task if running."""
    global _task
    if _task and not _task.done():
        _task.cancel()
        print(f"[{datetime.now()}] Keep-alive task stopped")
        _task = None

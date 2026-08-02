from __future__ import annotations

import asyncio
from typing import Any, Optional, Set

from fastapi import WebSocket


class WebSocketManager:
    """Broadcasts JSON events to connected clients. Thread-safe for
    publishing from background (non-asyncio) threads via `loop`.
    """

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def _broadcast(self, event: str, data: Any) -> None:
        message = {"event": event, "data": data}
        dead = []
        for client in list(self._clients):
            try:
                await client.send_json(message)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    def publish(self, event: str, data: Any) -> None:
        """Safe to call from any thread once the loop is bound."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(event, data), self._loop)


ws_manager = WebSocketManager()

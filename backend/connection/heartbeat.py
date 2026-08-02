from __future__ import annotations

import time


class HeartbeatMonitor:
    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout
        self._last_beat: float | None = None

    def beat(self) -> None:
        self._last_beat = time.monotonic()

    def is_alive(self) -> bool:
        if self._last_beat is None:
            return False
        return (time.monotonic() - self._last_beat) < self.timeout

    def reset(self) -> None:
        self._last_beat = None

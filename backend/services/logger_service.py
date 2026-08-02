from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Deque, List

MAX_LOG_ENTRIES = 500


@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str


class LoggerService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Deque[LogEntry] = deque(maxlen=MAX_LOG_ENTRIES)
        self._on_entry: Callable[[LogEntry], None] | None = None

    def set_listener(self, listener: Callable[[LogEntry], None]) -> None:
        self._on_entry = listener

    def log(self, message: str, level: str = "info") -> LogEntry:
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level,
            message=message,
        )
        with self._lock:
            self._entries.append(entry)
        if self._on_entry:
            self._on_entry(entry)
        return entry

    def get_entries(self) -> List[dict]:
        with self._lock:
            return [asdict(e) for e in self._entries]


logger_service = LoggerService()

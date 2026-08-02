from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional


@dataclass
class TelemetryState:
    connected: bool = False
    armed: bool = False
    mode: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading: Optional[float] = None
    ground_speed: Optional[float] = None
    battery_voltage: Optional[float] = None
    battery_remaining: Optional[int] = None
    gps_fix: Optional[str] = None
    satellite_count: Optional[int] = None
    heartbeat_status: str = "lost"


class TelemetryService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = TelemetryState()
        self._on_update: Callable[[dict], None] | None = None

    def set_listener(self, listener: Callable[[dict], None]) -> None:
        self._on_update = listener

    def update(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            snapshot = asdict(self._state)
        if self._on_update:
            self._on_update(snapshot)

    def reset(self) -> None:
        with self._lock:
            self._state = TelemetryState()
            snapshot = asdict(self._state)
        if self._on_update:
            self._on_update(snapshot)

    def get_state(self) -> dict:
        with self._lock:
            return asdict(self._state)


telemetry_service = TelemetryService()

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import List, Optional

from serial.tools import list_ports

from connection.heartbeat import HeartbeatMonitor
from mavlink.client import MavlinkClient
from mavlink.parser import handle_message
from services.logger_service import logger_service
from services.telemetry_service import telemetry_service

RECONNECT_DELAY_SECONDS = 2.0
HEARTBEAT_TIMEOUT_SECONDS = 3.0


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LOST = "lost"
    RECONNECTING = "reconnecting"


class ConnectionManager:
    def __init__(self) -> None:
        self.client = MavlinkClient()
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._device: Optional[str] = None
        self._baud: int = 57600
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._manual_disconnect = False
        self._heartbeat = HeartbeatMonitor(timeout=HEARTBEAT_TIMEOUT_SECONDS)
        self._on_state_change = None

    def set_state_listener(self, listener) -> None:
        self._on_state_change = listener

    @property
    def state(self) -> ConnectionState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: ConnectionState) -> None:
        with self._state_lock:
            self._state = state
        if self._on_state_change:
            self._on_state_change(state.value)

    @staticmethod
    def list_ports() -> List[dict]:
        return [
            {"device": p.device, "description": p.description}
            for p in list_ports.comports()
        ]

    def connect(self, device: str, baud: int = 57600) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._device = device
        self._baud = baud
        self._manual_disconnect = False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._manual_disconnect = True
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.client.close()
        self._heartbeat.reset()
        telemetry_service.reset()
        self._set_state(ConnectionState.DISCONNECTED)
        logger_service.log("Disconnected")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            connected = self._try_connect()
            if not connected:
                if self._manual_disconnect:
                    return
                self._set_state(ConnectionState.RECONNECTING)
                if self._stop_event.wait(RECONNECT_DELAY_SECONDS):
                    return
                continue

            self._read_loop()

            if self._manual_disconnect or self._stop_event.is_set():
                return

            self._set_state(ConnectionState.LOST)
            logger_service.log("Heartbeat Lost", level="warning")
            telemetry_service.update(connected=True, heartbeat_status="lost")
            self._set_state(ConnectionState.RECONNECTING)
            self.client.close()
            if self._stop_event.wait(RECONNECT_DELAY_SECONDS):
                return

    def _try_connect(self) -> bool:
        self._set_state(ConnectionState.CONNECTING)
        try:
            self.client.connect(self._device, self._baud)
            if not self.client.wait_heartbeat(timeout=10.0):
                self.client.close()
                return False
        except Exception as exc:
            logger_service.log(f"Connection error: {exc}", level="error")
            self.client.close()
            return False

        self.client.request_data_streams()
        self._heartbeat.beat()
        self._set_state(ConnectionState.CONNECTED)
        telemetry_service.update(connected=True, heartbeat_status="alive")
        logger_service.log("Connected")
        return True

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            msg = self.client.recv_match(blocking=False, timeout=0.5)
            if msg is not None:
                if msg.get_type() == "HEARTBEAT":
                    self._heartbeat.beat()
                handle_message(msg)
            else:
                time.sleep(0.05)

            if not self._heartbeat.is_alive():
                return


connection_manager = ConnectionManager()

from __future__ import annotations

from typing import Optional

from pymavlink import mavutil

HEARTBEAT_TIMEOUT_SECONDS = 3.0
MODE_MAP_ROVER = {
    "MANUAL": 0,
    "HOLD": 4,
    "AUTO": 10,
    "RTL": 11,
}


class MavlinkClient:
    """Thin wrapper around a pymavlink connection. No business logic —
    just reading/sending packets and mapping mode names to numbers.
    """

    def __init__(self) -> None:
        self._conn: Optional[mavutil.mavfile] = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def connect(self, device: str, baud: int = 57600) -> None:
        self._conn = mavutil.mavlink_connection(device, baud=baud)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def wait_heartbeat(self, timeout: float = 10.0) -> bool:
        if self._conn is None:
            return False
        msg = self._conn.wait_heartbeat(timeout=timeout)
        return msg is not None

    def recv_match(self, blocking: bool = False, timeout: float = 0.5):
        if self._conn is None:
            return None
        return self._conn.recv_match(blocking=blocking, timeout=timeout)

    def target_ids(self):
        if self._conn is None:
            return (0, 0)
        return self._conn.target_system, self._conn.target_component

    def arm(self, arm: bool) -> None:
        if self._conn is None:
            return
        sysid, compid = self.target_ids()
        self._conn.mav.command_long_send(
            sysid,
            compid,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1 if arm else 0,
            0, 0, 0, 0, 0, 0,
        )

    def set_mode(self, mode_name: str) -> bool:
        if self._conn is None:
            return False
        mode_map = self._conn.mode_mapping() or MODE_MAP_ROVER
        mode_name = mode_name.upper()
        if mode_name not in mode_map:
            return False
        mode_id = mode_map[mode_name]
        self._conn.mav.set_mode_send(
            self._conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        return True

    def reboot(self) -> None:
        if self._conn is None:
            return
        sysid, compid = self.target_ids()
        self._conn.mav.command_long_send(
            sysid,
            compid,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def request_data_streams(self, rate_hz: int = 4) -> None:
        if self._conn is None:
            return
        sysid, compid = self.target_ids()
        self._conn.mav.request_data_stream_send(
            sysid,
            compid,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            rate_hz,
            1,
        )

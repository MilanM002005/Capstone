from __future__ import annotations

from connection.serial_manager import ConnectionState, connection_manager
from services.logger_service import logger_service


class VehicleServiceError(Exception):
    pass


def _ensure_connected() -> None:
    if connection_manager.state != ConnectionState.CONNECTED:
        raise VehicleServiceError("Vehicle is not connected")


def arm() -> None:
    _ensure_connected()
    connection_manager.client.arm(True)
    logger_service.log("Vehicle Armed")


def disarm() -> None:
    _ensure_connected()
    connection_manager.client.arm(False)
    logger_service.log("Vehicle Disarmed")


def set_mode(mode: str) -> None:
    _ensure_connected()
    ok = connection_manager.client.set_mode(mode)
    if not ok:
        raise VehicleServiceError(f"Unknown mode: {mode}")
    logger_service.log(f"Mode Changed -> {mode.upper()}")


def reboot() -> None:
    _ensure_connected()
    connection_manager.client.reboot()
    logger_service.log("Flight controller reboot requested")

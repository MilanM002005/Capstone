from __future__ import annotations

from pymavlink import mavutil

from services.telemetry_service import telemetry_service

GPS_FIX_TYPES = {
    0: "NO_GPS",
    1: "NO_FIX",
    2: "2D_FIX",
    3: "3D_FIX",
    4: "DGPS",
    5: "RTK_FLOAT",
    6: "RTK_FIXED",
}


def handle_message(msg) -> None:
    """Update telemetry state from a single MAVLink message. No side
    effects beyond writing into the telemetry service.
    """
    msg_type = msg.get_type()

    if msg_type == "HEARTBEAT":
        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        mode = mavutil.mode_string_v10(msg)
        telemetry_service.update(armed=armed, mode=mode, heartbeat_status="alive")

    elif msg_type == "GLOBAL_POSITION_INT":
        telemetry_service.update(
            latitude=msg.lat / 1e7,
            longitude=msg.lon / 1e7,
            heading=msg.hdg / 100 if msg.hdg != 65535 else None,
        )

    elif msg_type == "VFR_HUD":
        telemetry_service.update(
            ground_speed=msg.groundspeed,
            heading=msg.heading,
        )

    elif msg_type == "SYS_STATUS":
        voltage = msg.voltage_battery / 1000 if msg.voltage_battery != 65535 else None
        remaining = msg.battery_remaining if msg.battery_remaining != -1 else None
        telemetry_service.update(
            battery_voltage=voltage,
            battery_remaining=remaining,
        )

    elif msg_type == "GPS_RAW_INT":
        telemetry_service.update(
            gps_fix=GPS_FIX_TYPES.get(msg.fix_type, "UNKNOWN"),
            satellite_count=msg.satellites_visible if msg.satellites_visible != 255 else None,
        )

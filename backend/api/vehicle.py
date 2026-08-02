from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from connection.serial_manager import connection_manager
from services import vehicle_service
from services.logger_service import logger_service
from services.telemetry_service import telemetry_service
from services.vehicle_service import VehicleServiceError

router = APIRouter()


class ConnectRequest(BaseModel):
    device: str
    baud: int = 57600


class ModeRequest(BaseModel):
    mode: str


@router.get("/vehicle")
def get_vehicle():
    telemetry = telemetry_service.get_state()
    return {
        "connectionState": connection_manager.state.value,
        "armed": telemetry["armed"],
        "mode": telemetry["mode"],
    }


@router.get("/ports")
def get_ports():
    return connection_manager.list_ports()


@router.post("/connect")
def connect(req: ConnectRequest):
    connection_manager.connect(req.device, req.baud)
    return {"status": "connecting"}


@router.post("/disconnect")
def disconnect():
    connection_manager.disconnect()
    return {"status": "disconnected"}


@router.post("/arm")
def arm():
    try:
        vehicle_service.arm()
    except VehicleServiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "ok"}


@router.post("/disarm")
def disarm():
    try:
        vehicle_service.disarm()
    except VehicleServiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "ok"}


@router.post("/mode")
def set_mode(req: ModeRequest):
    try:
        vehicle_service.set_mode(req.mode)
    except VehicleServiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "ok"}


@router.get("/logs")
def get_logs():
    return logger_service.get_entries()

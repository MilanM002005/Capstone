from __future__ import annotations

from fastapi import APIRouter

from services.telemetry_service import telemetry_service

router = APIRouter()


@router.get("/telemetry")
def get_telemetry():
    return telemetry_service.get_state()

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.telemetry import router as telemetry_router
from api.vehicle import router as vehicle_router
from connection.serial_manager import connection_manager
from services.logger_service import logger_service
from services.telemetry_service import telemetry_service
from ws.manager import ws_manager

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app = FastAPI(title="GCS Phase 1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telemetry_router, prefix="/api")
app.include_router(vehicle_router, prefix="/api")

if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.on_event("startup")
async def on_startup() -> None:
    loop = asyncio.get_running_loop()
    ws_manager.bind_loop(loop)
    telemetry_service.set_listener(lambda state: ws_manager.publish("telemetry", state))
    logger_service.set_listener(lambda entry: ws_manager.publish("log", entry.__dict__))
    connection_manager.set_state_listener(lambda state: ws_manager.publish("connection", {"state": state}))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    connection_manager.disconnect()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# Serves the built React app. Static files copied from frontend/public
# (favicon, splash image, etc.) are served as-is; any other path falls
# back to index.html so client-side routing (react-router) can handle it.
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str) -> FileResponse:
    requested = (FRONTEND_DIST / full_path).resolve()
    if requested.is_file() and FRONTEND_DIST.resolve() in requested.parents:
        return FileResponse(requested)

    index_path = FRONTEND_DIST / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend build not found — run `npm run build`")
    return FileResponse(index_path)

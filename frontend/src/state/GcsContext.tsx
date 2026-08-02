import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import { gcsWebSocket } from "../services/websocket";
import type { ConnectionState, LogEntry, TelemetryState } from "../types";

interface GcsContextValue {
  telemetry: TelemetryState;
  connectionState: ConnectionState;
  logs: LogEntry[];
}

const initialTelemetry: TelemetryState = {
  connected: false,
  armed: false,
  mode: null,
  latitude: null,
  longitude: null,
  heading: null,
  ground_speed: null,
  battery_voltage: null,
  battery_remaining: null,
  gps_fix: null,
  satellite_count: null,
  heartbeat_status: "lost",
};

const GcsContext = createContext<GcsContextValue | null>(null);

export function GcsProvider({ children }: { children: ReactNode }) {
  const [telemetry, setTelemetry] = useState<TelemetryState>(initialTelemetry);
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [logs, setLogs] = useState<LogEntry[]>([]);

  useEffect(() => {
    api.getTelemetry().then(setTelemetry).catch(() => undefined);
    api.getVehicle().then((v) => setConnectionState(v.connectionState)).catch(() => undefined);
    api.getLogs().then(setLogs).catch(() => undefined);

    gcsWebSocket.connect();
    const offTelemetry = gcsWebSocket.on("telemetry", (data) => setTelemetry(data as TelemetryState));
    const offConnection = gcsWebSocket.on("connection", (data) =>
      setConnectionState((data as { state: ConnectionState }).state)
    );
    const offLog = gcsWebSocket.on("log", (data) =>
      setLogs((prev) => [...prev.slice(-499), data as LogEntry])
    );

    return () => {
      offTelemetry();
      offConnection();
      offLog();
      gcsWebSocket.disconnect();
    };
  }, []);

  const value = useMemo(
    () => ({ telemetry, connectionState, logs }),
    [telemetry, connectionState, logs]
  );

  return <GcsContext.Provider value={value}>{children}</GcsContext.Provider>;
}

export function useGcs(): GcsContextValue {
  const ctx = useContext(GcsContext);
  if (!ctx) {
    throw new Error("useGcs must be used within a GcsProvider");
  }
  return ctx;
}

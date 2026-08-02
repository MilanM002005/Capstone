import type { LogEntry, SerialPort, TelemetryState, VehicleStatus } from "../types";

const BASE_URL = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  getTelemetry: () => request<TelemetryState>("/telemetry"),
  getVehicle: () => request<VehicleStatus>("/vehicle"),
  getPorts: () => request<SerialPort[]>("/ports"),
  getLogs: () => request<LogEntry[]>("/logs"),
  connect: (device: string, baud = 57600) =>
    request<{ status: string }>("/connect", {
      method: "POST",
      body: JSON.stringify({ device, baud }),
    }),
  disconnect: () => request<{ status: string }>("/disconnect", { method: "POST" }),
  arm: () => request<{ status: string }>("/arm", { method: "POST" }),
  disarm: () => request<{ status: string }>("/disarm", { method: "POST" }),
  setMode: (mode: string) =>
    request<{ status: string }>("/mode", {
      method: "POST",
      body: JSON.stringify({ mode }),
    }),
};

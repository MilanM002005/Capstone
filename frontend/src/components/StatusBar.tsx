import type { ConnectionState } from "../types";

const LABELS: Record<ConnectionState, string> = {
  disconnected: "Disconnected",
  connecting: "Connecting…",
  connected: "Connected",
  lost: "Connection Lost",
  reconnecting: "Reconnecting…",
};

const COLORS: Record<ConnectionState, string> = {
  disconnected: "#9ca3af",
  connecting: "#f59e0b",
  connected: "#22c55e",
  lost: "#ef4444",
  reconnecting: "#f59e0b",
};

export function StatusBar({ state }: { state: ConnectionState }) {
  return (
    <div className="status-bar">
      <span className="status-dot" style={{ backgroundColor: COLORS[state] }} />
      <span>{LABELS[state]}</span>
    </div>
  );
}

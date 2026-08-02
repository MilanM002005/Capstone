import { useEffect, useState } from "react";
import type { ConnectionState } from "../types";
import { ThemeToggle } from "./ThemeToggle";

const CONNECTION_LABELS: Record<ConnectionState, string> = {
  disconnected: "Disconnected",
  connecting: "Connecting",
  connected: "Connected",
  lost: "Connection Lost",
  reconnecting: "Reconnecting",
};

const CONNECTION_COLORS: Record<ConnectionState, string> = {
  disconnected: "#8a94a6",
  connecting: "#e5a83d",
  connected: "#4caf50",
  lost: "#e53935",
  reconnecting: "#e5a83d",
};

interface GcsHeaderProps {
  missionStatus: string;
  connectionState: ConnectionState;
}

export function GcsHeader({ missionStatus, connectionState }: GcsHeaderProps) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="gcs-header">
      <div className="gcs-header-status">Mission Status: {missionStatus}</div>
      <div className="gcs-header-title">Autonomous Ground Control Station</div>
      <div className="gcs-header-right">
        <span className="gcs-header-clock">{now.toLocaleTimeString()}</span>
        <span className="gcs-connection-pill">
          <span
            className="status-dot"
            style={{ backgroundColor: CONNECTION_COLORS[connectionState] }}
          />
          {CONNECTION_LABELS[connectionState]}
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}

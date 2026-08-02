import type { ConnectionState, TelemetryState } from "../types";

interface BottomStatusBarProps {
  telemetry: TelemetryState;
  connectionState: ConnectionState;
}

function formatValue(value: string | number | null | undefined, suffix = ""): string {
  if (value === null || value === undefined) return "--";
  return `${value}${suffix}`;
}

export function BottomStatusBar({ telemetry, connectionState }: BottomStatusBarProps) {
  const gps =
    telemetry.gps_fix && telemetry.satellite_count !== null
      ? `${telemetry.gps_fix} (${telemetry.satellite_count} sats)`
      : "--";

  const battery =
    telemetry.battery_voltage !== null
      ? `${telemetry.battery_voltage.toFixed(1)}V${
          telemetry.battery_remaining !== null ? ` / ${telemetry.battery_remaining}%` : ""
        }`
      : "--";

  return (
    <footer className="gcs-status-bar">
      <span className="status-item">
        <span className="status-label">GPS:</span>
        <span className="status-value">{gps}</span>
      </span>
      <span className="status-item">
        <span className="status-label">Mode:</span>
        <span className="status-value">{formatValue(telemetry.mode)}</span>
      </span>
      <span className="status-item">
        <span className="status-label">Battery:</span>
        <span className="status-value">{battery}</span>
      </span>
      <span className="status-item">
        <span className="status-label">Heading:</span>
        <span className="status-value">
          {telemetry.heading !== null ? formatValue(telemetry.heading.toFixed(0), "°") : "--"}
        </span>
      </span>
      <span className="status-item">
        <span className="status-label">Speed:</span>
        <span className="status-value">
          {telemetry.ground_speed !== null ? formatValue(telemetry.ground_speed.toFixed(1), " m/s") : "--"}
        </span>
      </span>
      <span className="status-item">
        <span className="status-label">Telemetry:</span>
        <span className="status-value">{telemetry.connected ? "Connected" : "Disconnected"}</span>
      </span>
      <span className="status-item">
        <span className="status-label">Connection:</span>
        <span className="status-value">{connectionState}</span>
      </span>
    </footer>
  );
}

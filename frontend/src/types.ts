export type ConnectionState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "lost"
  | "reconnecting";

export interface TelemetryState {
  connected: boolean;
  armed: boolean;
  mode: string | null;
  latitude: number | null;
  longitude: number | null;
  heading: number | null;
  ground_speed: number | null;
  battery_voltage: number | null;
  battery_remaining: number | null;
  gps_fix: string | null;
  satellite_count: number | null;
  heartbeat_status: "alive" | "lost";
}

export interface VehicleStatus {
  connectionState: ConnectionState;
  armed: boolean;
  mode: string | null;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

export interface SerialPort {
  device: string;
  description: string;
}

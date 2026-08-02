import { useGcs } from "../state/GcsContext";

export function Telemetry() {
  const { telemetry } = useGcs();

  const rows: [string, string][] = [
    ["Latitude", telemetry.latitude?.toFixed(6) ?? "—"],
    ["Longitude", telemetry.longitude?.toFixed(6) ?? "—"],
    ["Heading", telemetry.heading !== null ? `${telemetry.heading.toFixed(0)}°` : "—"],
    ["Ground Speed", telemetry.ground_speed !== null ? `${telemetry.ground_speed.toFixed(1)} m/s` : "—"],
    ["Battery", telemetry.battery_voltage !== null ? `${telemetry.battery_voltage.toFixed(2)} V` : "—"],
    ["GPS Status", telemetry.gps_fix ?? "—"],
    ["Satellite Count", telemetry.satellite_count?.toString() ?? "—"],
    ["Heartbeat", telemetry.heartbeat_status],
  ];

  return (
    <div className="page">
      <h1>Telemetry</h1>
      <div className="card-grid">
        {rows.map(([label, value]) => (
          <div className="card" key={label}>
            <h3>{label}</h3>
            <p>{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
